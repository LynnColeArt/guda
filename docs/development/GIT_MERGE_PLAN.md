# Git History Merge Plan

## Goal
Preserve gonum's optimization history while making it part of GUDA's development story.

## Steps

### 1. Initialize GUDA Repository
```bash
git init
git add .
git commit -m "GUDA: Pre-assimilation checkpoint"
```

### 2. Add Gonum as Subtree
```bash
# Add gonum as a remote
git remote add gonum-history ./gonum
git fetch gonum-history

# Merge histories
git merge --allow-unrelated-histories gonum-history/master -m "Assimilate gonum compute engine into GUDA"
```

### 3. Mark the Assimilation
```bash
git tag v0.1.0-assimilation -m "The point where GUDA assimilated gonum"
```

### 4. Document the Transition
```bash
git add compute/ ASSIMILATION_*.md
git commit -m "feat: Complete gonum assimilation into native compute engine

- Moved BLAS implementations to compute/
- Removed complex number support
- Prepared for Float32 AVX2 optimizations
- Fixed critical matrix multiplication bugs"
```

## Benefits
- Full history of optimizations
- Can trace back any numerical algorithms
- Shows the evolution from external dependency to integrated engine
- Preserves attribution and licenses

## Alternative: Keep Separate
If we want to be more conservative:
1. Keep gonum/.git as is
2. Init GUDA git
3. Reference gonum as a submodule at a specific commit
4. But then gradually replace submodule files with compute/

What do you think?