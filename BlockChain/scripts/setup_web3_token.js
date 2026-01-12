const fs = require('fs');
const path = require('path');
const readline = require('readline');
const { exec } = require('child_process');

const projectRoot = path.resolve(__dirname, '..');
const envPath = path.join(projectRoot, '.env');
const envExamplePath = path.join(projectRoot, 'env.example');
const tokenUrl = 'https://app.pinata.cloud/keys';

function openUrlInBrowser(url) {
  const platform = process.platform;
  let cmd = '';
  if (platform === 'darwin') cmd = `open "${url}"`;
  else if (platform === 'win32') cmd = `start "" "${url}"`;
  else cmd = `xdg-open "${url}"`;
  exec(cmd, (err) => {
    if (err) {
      console.warn('无法自动打开浏览器，请手动访问:', url);
    }
  });
}

function ensureEnvBase() {
  if (!fs.existsSync(envPath)) {
    if (fs.existsSync(envExamplePath)) {
      fs.copyFileSync(envExamplePath, envPath);
      console.log('已创建 .env（从 env.example 复制）。');
    } else {
      fs.writeFileSync(envPath, '', { encoding: 'utf8' });
      console.log('已创建空的 .env。');
    }
  }
}

function writePinataToEnv({ jwt, apiKey, apiSecret }) {
  ensureEnvBase();
  const raw = fs.readFileSync(envPath, 'utf8');
  const lines = raw.length ? raw.split(/\r?\n/) : [];
  let hasJwt = false;
  let hasApiKey = false;
  let hasApiSecret = false;
  const updated = lines.map((line) => {
    if (/^\s*PINATA_JWT\s*=/.test(line)) {
      hasJwt = true;
      return `PINATA_JWT=${jwt || ''}`;
    }
    if (/^\s*PINATA_API_KEY\s*=/.test(line)) {
      hasApiKey = true;
      return `PINATA_API_KEY=${apiKey || ''}`;
    }
    if (/^\s*PINATA_API_SECRET\s*=/.test(line)) {
      hasApiSecret = true;
      return `PINATA_API_SECRET=${apiSecret || ''}`;
    }
    return line;
  });
  if (!hasJwt && jwt) updated.push(`PINATA_JWT=${jwt}`);
  if (!hasApiKey && apiKey) updated.push(`PINATA_API_KEY=${apiKey}`);
  if (!hasApiSecret && apiSecret) updated.push(`PINATA_API_SECRET=${apiSecret}`);
  if (updated.length === 0 || updated[updated.length - 1] !== '') updated.push('');
  fs.writeFileSync(envPath, updated.join('\n'), { encoding: 'utf8' });
  console.log('已写入 .env: PINATA_JWT / PINATA_API_KEY / PINATA_API_SECRET');
}

function parseArgValue(name, short) {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === `--${name}` || arg === `-${short}`) {
      return args[i + 1] || '';
    }
    if (arg.startsWith(`--${name}=`)) {
      return arg.split('=').slice(1).join('=');
    }
  }
  return '';
}

async function promptPinataInteractive() {
  return await new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question('请输入 Pinata JWT（留空则输入 API Key/Secret）:\nJWT: ', (jwtAns) => {
      const jwt = (jwtAns || '').trim();
      if (jwt) {
        rl.close();
        return resolve({ jwt });
      }
      rl.question('API Key: ', (keyAns) => {
        rl.question('API Secret: ', (secAns) => {
          rl.close();
          resolve({ apiKey: (keyAns || '').trim(), apiSecret: (secAns || '').trim() });
        });
      });
    });
  });
}

async function main() {
  const jwtFromArg = parseArgValue('jwt', 'j');
  const apiKeyFromArg = parseArgValue('apiKey', 'k');
  const apiSecretFromArg = parseArgValue('apiSecret', 's');
  if (jwtFromArg || (apiKeyFromArg && apiSecretFromArg)) {
    writePinataToEnv({ jwt: jwtFromArg, apiKey: apiKeyFromArg, apiSecret: apiSecretFromArg });
    return;
  }

  console.log('将为你打开 Pinata 的 API Keys 页面：');
  console.log(tokenUrl);
  openUrlInBrowser(tokenUrl);

  if (process.stdin.isTTY) {
    const creds = await promptPinataInteractive();
    if (!creds || (!creds.jwt && !(creds.apiKey && creds.apiSecret))) {
      console.warn('未输入 Pinata 凭据。也可以使用命令行参数: --jwt 或 --apiKey/--apiSecret');
      return;
    }
    writePinataToEnv(creds);
  } else {
    console.log('非交互环境。请在浏览器创建后运行以下之一:');
    console.log('node scripts/setup_web3_token.js --jwt=你的JWT');
    console.log('node scripts/setup_web3_token.js --apiKey=你的Key --apiSecret=你的Secret');
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});


