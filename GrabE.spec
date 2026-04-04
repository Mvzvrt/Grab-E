# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules
import glob
import os

# Data files: model and diagram folder
datas = [
    ('mgc_core/third_party/sed/model.yml.gz', 'mgc_core/third_party/sed'),
    ('diagram', 'diagram'),
    # Bundle all UI assets (icons, splash artwork, and how-to screenshots).
    ('src/public', 'public'),
    # Include root-level Python modules needed by src
    ('color_space.py', '.'),
    ('mgc_api.py', '.'),
    ('grabcut.py', '.'),
    ('io_utils.py', '.'),
]

# Binary files: compiled C extension
binaries = []
# Find all .pyd files in mgc_core
for pyd in glob.glob('mgc_core/*.pyd'):
    binaries.append((pyd, 'mgc_core'))

# Hidden imports
hiddenimports = [
    'mgc_core.fastgeo',
    'mgc_core.core',
    'cv2',
    'cv2.ximgproc',
    'numpy',
    'PIL',
    'skimage',
]

# Collect all submodules for complex packages
hiddenimports += collect_submodules('cv2')
hiddenimports += collect_submodules('skimage')

# Collect PySide6
tmp_ret = collect_all('PySide6')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Collect numpy (for mkl/openblas DLLs)
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]


a = Analysis(
    ['src/main.py'],
    pathex=['.', 'src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GrabE',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for windowed mode (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GrabE',
)
