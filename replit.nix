
{ pkgs }: {
  deps = [
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.cairo
    pkgs.xsimd
    pkgs.pkg-config
    pkgs.libxcrypt
    pkgs.ffmpeg-full
    pkgs.glibcLocales
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nodePackages.typescript-language-server
    pkgs.nodejs
    pkgs.yarn
    pkgs.streamlit
  ];
}
