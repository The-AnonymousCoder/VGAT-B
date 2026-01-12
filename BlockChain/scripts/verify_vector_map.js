require('dotenv').config();
const fs = require('fs');
const path = require('path');
const { ethers } = require('hardhat');

function getArg(name, short) {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === `--${name}` || a === `-${short}`) return args[i + 1] || '';
    if (a.startsWith(`--${name}=`)) return a.split('=').slice(1).join('=');
  }
  return '';
}

async function main() {
  const addressFromArg = getArg('address', 'a');
  const pathFromArg = getArg('path', 'p');
  const cfgPath = getArg('config', 'c');

  let logicalPath = pathFromArg;
  if (!logicalPath && cfgPath) {
    const cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    logicalPath = cfg.logicalPath || `${cfg.bucket}/${cfg.basePath}`;
  }

  const address = addressFromArg || process.env.CONTRACT_ADDRESS;
  if (!address) {
    console.error('Missing contract address. Use --address or set CONTRACT_ADDRESS in .env');
    process.exit(1);
  }
  if (!logicalPath) {
    console.error('Missing logical path. Use --path or provide --config pointing to oss.config.json');
    process.exit(1);
  }

  const registry = await ethers.getContractAt('VectorMapRegistry', address);
  const cid = await registry.getManifestCID(logicalPath);
  console.log('LogicalPath:', logicalPath);
  console.log('Contract   :', address);
  console.log('ManifestCID:', cid || '(empty)');
  if (cid) {
    console.log('Gateway    :', `https://ipfs.io/ipfs/${cid}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});


