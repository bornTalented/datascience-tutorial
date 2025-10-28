To get the **tree structure** of directories and files using the terminal, you can use the `tree` command.

---

### ✅ Option 1: **Using `tree` command (Recommended)**

Most Unix-like systems (Linux, macOS) support the `tree` command, but it might not be installed by default.

#### **1. Install `tree`** (if not already installed)

- **Ubuntu/Debian:**
    
    ```bash
    sudo apt install tree
    ```
    
- **macOS (using Homebrew):**
    
    ```bash
    brew install tree
    ```
    
- **Fedora:**
    
    ```bash
    sudo dnf install tree
    ```
    

---

#### **2. Use `tree`**

- To show the structure of the current directory:
    
    ```bash
    tree
    ```
    
- To show hidden files as well:
    
    ```bash
    tree -a
    ```
    
- To limit depth (e.g., only 2 levels deep):
    
    ```bash
    tree -L 2
    ```
    
- To print full paths:
    
    ```bash
    tree -f
    ```
    

---

### 🔁 Option 2: **Without `tree` (using `find` command)**

If you don't want to or can't install `tree`, use:

```bash
find . -print | sed -e 's;[^/]*/;|____;g;s;____|; |;g'
```

This gives a rough tree-like structure using `find`.

---

### 📝 Example output using `tree`:

```
.
├── file1.txt
├── dir1
│   ├── file2.txt
│   └── file3.txt
└── dir2
    └── file4.txt
```

Yes, you **can exclude specific directories** when using the `tree` command by using the `-I` (uppercase i) option to specify patterns to **ignore**.

---

### 🔹 **Syntax**

```bash
tree -I 'pattern'
```

- `pattern` can be a directory or file name (supports wildcards and multiple patterns with `|`).
    
- It's a **case-sensitive** match against names (not full paths).
    

---

### 🔹 **Examples**

#### ✅ Exclude one directory:

```bash
tree -I 'node_modules'
```

#### ✅ Exclude multiple directories:

```bash
tree -I 'node_modules|dist|.git'
```

#### ✅ Exclude directories and files:

```bash
tree -I 'node_modules|*.log|dist'
```

---

### 📝 Notes:

- This only **matches names**, not full paths. So if you have multiple `dist` folders at different levels, **all** will be excluded.
    
- You can still combine this with other flags like `-L` for depth, `-a` for hidden files, etc.
    

---

### ✅ Example:

```bash
tree -I 'venv|__pycache__|.git' -L 2
```

This will show your project tree up to depth 2, ignoring common Python virtual environment and cache folders.
