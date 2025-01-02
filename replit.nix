
{ pkgs }: {
  deps = [
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
