
{ pkgs }: {
  deps = [
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
