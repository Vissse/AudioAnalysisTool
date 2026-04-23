# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['C:\\Users\\Vit\\Documents\\AudioAnalysisTool\\audio_matcher\\main.py'],
    pathex=[],
    binaries=[],
    datas=[('C:\\Users\\Vit\\Documents\\AudioAnalysisTool\\assets', 'assets'), ('C:\\Users\\Vit\\Documents\\AudioAnalysisTool\\data\\processed', 'data/processed'), ('C:\\Users\\Vit\\Documents\\AudioAnalysisTool\\models', 'models')],
    hiddenimports=[],
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
    name='AudioAnalysisTool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\Vit\\Documents\\AudioAnalysisTool\\assets\\ikona.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AudioAnalysisTool',
)
