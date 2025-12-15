# Frontend Installation Fix

## Network Connection Issues

If you're getting `ECONNRESET` errors, try these solutions:

### Solution 1: Retry with Clean Cache
```bash
npm cache clean --force
npm install
```

### Solution 2: Use Different Registry
```bash
npm install --registry https://registry.npmjs.org/
```

### Solution 3: Install with Retry
```bash
npm install --fetch-retries=5 --fetch-retry-mintimeout=20000
```

### Solution 4: Use Yarn (Alternative Package Manager)
```bash
# Install yarn first
npm install -g yarn

# Then use yarn
yarn install
yarn dev
```

### Solution 5: Manual Installation (If network keeps failing)
If npm continues to fail, you can manually download and install dependencies:

1. Create `node_modules` folder if it doesn't exist
2. Download packages manually or use a different network
3. Or use the standalone HTML version (see below)

## Alternative: Standalone HTML Version

If npm installation continues to fail, I can create a standalone HTML/JS version that doesn't require npm. Let me know if you need this.

