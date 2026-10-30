# Lab + headless browser on the pod, so drives can be recorded there (when uploading frames to the pod is slow).
# Expects /root/labcode.tgz = src/ + slam/record_drive.mjs + slam/package.json from the working copy.
set -e
curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar xJ -C /opt
export PATH=/opt/node-v20.18.0-linux-x64/bin:$PATH
[ -d /root/lab ] || git clone -q https://github.com/sadeeqgandalf/lunar-occlusion-tracking /root/lab
cd /root/lab && tar xzf /root/labcode.tgz
cd slam && npm i --silent && npx playwright-core install --with-deps chromium > /root/pw.log 2>&1 && cd ..
(nohup node server.js > /root/server.log 2>&1 &)
echo LAB_READY
