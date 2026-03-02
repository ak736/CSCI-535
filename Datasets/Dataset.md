````markdown
# AMI Meeting Corpus – Download Instructions

This guide explains how to download the **AMI Meeting Corpus** dataset using the provided `dataset.sh` script.

---

## Prerequisites

- A Unix-based system (Linux, macOS, or WSL on Windows)
- `wget` installed

Check if `wget` is available:

```bash
wget --version
```
````

If not installed:

**Ubuntu / Debian**

```bash
sudo apt update && sudo apt install wget
```

**macOS (Homebrew)**

```bash
brew install wget
```

---

## Download Instructions

**1. Make the script executable**

```bash
chmod +x dataset.sh
```

**2. Run the script**

```bash
./dataset.sh
```

The script will automatically download the AMI Meeting Corpus into the expected directory structure.

---

## Notes

- Ensure you have sufficient disk space before starting the download.
- The download may take some time depending on your internet connection.
- Downloaded files will be placed in `data/raw/` by default.

```

```
