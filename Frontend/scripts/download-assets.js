const fs = require('fs');
const path = require('path');
const https = require('https');

const assets = {
  'bootstrap.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'bootstrap.bundle.min.js': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js',
  'Poppins-Regular.woff2': 'https://fonts.gstatic.com/s/poppins/v20/pxiEyp8kv8JHgFVrJJfecg.woff2'
};

const downloadFile = (url, destination) => {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destination);
    https.get(url, (response) => {
      response.pipe(file);
      file.on('finish', () => {
        file.close();
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(destination, () => {});
      reject(err);
    });
  });
};

const downloadAssets = async () => {
  const baseDir = path.join(__dirname, '..', 'public', 'assets');
  
  // Create directories if they don't exist
  ['css', 'js', 'fonts'].forEach(dir => {
    const dirPath = path.join(baseDir, dir);
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  });

  // Download CSS files
  await downloadFile(
    assets['bootstrap.min.css'],
    path.join(baseDir, 'css', 'bootstrap.min.css')
  );

  // Download JS files
  await downloadFile(
    assets['bootstrap.bundle.min.js'],
    path.join(baseDir, 'js', 'bootstrap.bundle.min.js')
  );

  // Download fonts
  await downloadFile(
    assets['Poppins-Regular.woff2'],
    path.join(baseDir, 'fonts', 'Poppins-Regular.woff2')
  );

  // Create font-face CSS
  const fontFaceCSS = `
@font-face {
  font-family: 'Poppins';
  font-style: normal;
  font-weight: 400;
  src: url('../fonts/Poppins-Regular.woff2') format('woff2');
}
  `;

  fs.writeFileSync(
    path.join(baseDir, 'css', 'fonts.css'),
    fontFaceCSS.trim()
  );
};

downloadAssets().catch(console.error); 