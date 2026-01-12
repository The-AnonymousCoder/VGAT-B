require('dotenv').config();
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const pinataSDK = require('@pinata/sdk');
const OSS = require('ali-oss');
const { ethers } = require('hardhat');

function sha256OfBuffer(buf) {
  const hash = crypto.createHash('sha256');
  hash.update(buf);
  return hash.digest('hex');
}

function listFilesSorted(dirPath) {
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  const files = entries.filter(e => e.isFile()).map(e => e.name);
  files.sort((a, b) => a.localeCompare(b));
  return files;
}

async function ensurePinata() {
  const jwt = process.env.PINATA_JWT;
  const apiKey = process.env.PINATA_API_KEY;
  const apiSecret = process.env.PINATA_API_SECRET;
  if (jwt) return new pinataSDK({ pinataJWTKey: jwt });
  if (apiKey && apiSecret) return new pinataSDK({ pinataApiKey: apiKey, pinataSecretApiKey: apiSecret });
  throw new Error('Missing Pinata credentials');
}

async function pinFile(pinata, filePath, name) {
  const stream = fs.createReadStream(filePath);
  const call = async () => {
    const res = await pinata.pinFileToIPFS(stream, {
      pinataMetadata: { name: name || path.basename(filePath) },
      pinataOptions: { cidVersion: 1 },
    });
    return res.IpfsHash;
  };
  return await withRetry(call, `pinFile ${name || path.basename(filePath)}`);
}

async function pinJSON(pinata, obj, name) {
  const call = async () => {
    const res = await pinata.pinJSONToIPFS(obj, {
      pinataMetadata: { name: name || 'manifest.json' },
      pinataOptions: { cidVersion: 1 },
    });
    return res.IpfsHash;
  };
  return await withRetry(call, `pinJSON ${name || 'manifest.json'}`);
}

async function withRetry(fn, label, attempts = 5, baseDelayMs = 800) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      return await fn();
    } catch (e) {
      lastErr = e;
      const delay = Math.round(baseDelayMs * Math.pow(2, i - 1));
      console.warn(`[retry ${i}/${attempts}] ${label} failed:`, e?.code || e?.message || e);
      if (i < attempts) await new Promise(r => setTimeout(r, Math.min(delay, 8000)));
    }
  }
  throw lastErr;
}

async function main() {
  // Two modes:
  // 1) Local mode: node scripts/upload_vector_map.js <logicalPath> <baseFolder> [originalDataSub]
  // 2) Config mode: node scripts/upload_vector_map.js --config ./oss.config.json
  let logicalPath, baseFolder, originalDataSub;
  const cfgFlagIdx = process.argv.indexOf('--config');
  if (cfgFlagIdx !== -1) {
    const cfgPath = process.argv[cfgFlagIdx + 1];
    if (!cfgPath) throw new Error('Missing --config path');
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    if (cfg.provider !== 'aliyun') throw new Error('Only provider=aliyun supported currently');
    const client = new OSS({
      endpoint: cfg.endpoint,
      accessKeyId: cfg.accessKeyId,
      accessKeySecret: cfg.accessKeySecret,
      bucket: cfg.bucket,
    });
    // Download OSS folder to local tmp dir
    const downloadDir = path.resolve(cfg.downloadDir || '.tmp_oss_download');
    if (!fs.existsSync(downloadDir)) fs.mkdirSync(downloadDir, { recursive: true });
    baseFolder = path.join(downloadDir, cfg.basePath);
    const originalDataLocalDir = path.join(baseFolder, cfg.originalDataSub || 'originalData');
    if (!fs.existsSync(originalDataLocalDir)) fs.mkdirSync(originalDataLocalDir, { recursive: true });

    // Helper to download a single object if exists
    async function downloadIfExists(objectKey, localFile) {
      try {
        const res = await client.head(objectKey);
        if (res && res.res && res.res.status === 200) {
          const dir = path.dirname(localFile);
          if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
          const r = await client.get(objectKey, localFile);
          if (!(r && r.res && r.res.status === 200)) throw new Error('Failed to download ' + objectKey);
        }
      } catch (e) {
        // ignore missing
      }
    }

    // List and download files in originalData
    const prefix = path.posix.join(cfg.basePath, cfg.originalDataSub || 'originalData') + '/';
    const files = [];
    let continuationToken = null;
    do {
      const result = await client.list({ prefix, marker: continuationToken }, {});
      continuationToken = result.nextMarker;
      for (const obj of result.objects || []) {
        if (obj.name.endsWith('/')) continue;
        files.push(obj.name);
      }
    } while (continuationToken);

    for (const objKey of files) {
      const rel = objKey.substring(prefix.length);
      const localPath = path.join(originalDataLocalDir, rel);
      await downloadIfExists(objKey, localPath);
    }

    // Download artifacts
    await downloadIfExists(path.posix.join(cfg.basePath, cfg.copyrightPng), path.join(baseFolder, cfg.copyrightPng));
    await downloadIfExists(path.posix.join(cfg.basePath, cfg.zeroWatermarkPng), path.join(baseFolder, cfg.zeroWatermarkPng));
    await downloadIfExists(path.posix.join(cfg.basePath, cfg.copyrightTxt), path.join(baseFolder, cfg.copyrightTxt));

    logicalPath = cfg.logicalPath || `${cfg.bucket}/${cfg.basePath}`;
    originalDataSub = cfg.originalDataSub || 'originalData';
  } else {
    logicalPath = process.argv[2];
    baseFolder = process.argv[3];
    originalDataSub = process.argv[4] || 'originalData';
    if (!logicalPath || !baseFolder) {
      console.log('Usage: node scripts/upload_vector_map.js <logicalPath> <baseFolder> [originalDataSub]');
      console.log('   or: node scripts/upload_vector_map.js --config ./oss.config.json');
      process.exit(1);
    }
  }

  const originalDataDir = path.join(baseFolder, originalDataSub);
  if (!fs.existsSync(originalDataDir)) throw new Error('originalData folder not found: ' + originalDataDir);

  // 1) Compute ordered hash over originalData: sort files, hash over filename + filecontent
  const fileNames = listFilesSorted(originalDataDir);
  if (fileNames.length === 0) throw new Error('No files in originalData');
  const hash = crypto.createHash('sha256');
  const vectorFiles = [];
  for (const name of fileNames) {
    const filePath = path.join(originalDataDir, name);
    const content = fs.readFileSync(filePath);
    hash.update(Buffer.from(name));
    hash.update(content);
    vectorFiles.push({ name, size: fs.statSync(filePath).size });
  }
  const orderedDigestHex = hash.digest('hex');

  // 2) Upload artifacts to IPFS: copyright, zeroWatermark, copyright.txt
  const pinata = await ensurePinata();
  const copyrightPng = path.join(baseFolder, 'copyright.png');
  const zeroWatermarkPng = path.join(baseFolder, 'zeroWatermark.png');
  const copyrightTxt = path.join(baseFolder, 'copyright.txt');

  let cidCopyright = '';
  let cidZeroWatermark = '';
  let cidCopyrightTxt = '';
  if (fs.existsSync(copyrightPng)) cidCopyright = await pinFile(pinata, copyrightPng, path.basename(copyrightPng));
  if (fs.existsSync(zeroWatermarkPng)) cidZeroWatermark = await pinFile(pinata, zeroWatermarkPng, path.basename(zeroWatermarkPng));
  if (fs.existsSync(copyrightTxt)) cidCopyrightTxt = await pinFile(pinata, copyrightTxt, path.basename(copyrightTxt));

  // 3) Build manifest JSON and pin to IPFS
  const manifest = {
    version: 1,
    logicalPath, // e.g. bucket/Institution/Railway
    orderedDigest: '0x' + orderedDigestHex,
    originalData: {
      baseFolder,
      folder: originalDataSub,
      files: vectorFiles,
      hashing: 'sha256(fileName + fileContent) with files sorted by name',
    },
    artifacts: {
      copyrightPng: cidCopyright,
      zeroWatermarkPng: cidZeroWatermark,
      copyrightTxt: cidCopyrightTxt,
    },
    createdAt: new Date().toISOString(),
  };
  const manifestCID = await pinJSON(pinata, manifest, `${path.basename(baseFolder)}-manifest.json`);
  console.log('Manifest CID:', manifestCID);

  // 4) Register on-chain: VectorMapRegistry.registerVectorMap(logicalPath, manifestCID)
  const CONTRACT = process.env.CONTRACT_ADDRESS;
  let registry;
  if (CONTRACT) {
    registry = await ethers.getContractAt('VectorMapRegistry', CONTRACT);
  } else {
    const Factory = await ethers.getContractFactory('VectorMapRegistry');
    registry = await Factory.deploy();
    await registry.waitForDeployment();
    const addr = await registry.getAddress();
    console.log('Deployed VectorMapRegistry at', addr);
  }
  const tx = await registry.registerVectorMap(logicalPath, manifestCID);
  const receipt = await tx.wait();
  console.log('Registered on-chain. Tx:', receipt.hash);
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});


