
{ pkgs }: {
  deps = [
    pkgs.ffmpeg-full
    pkgs.glibcLocales
    pkgs.python311
    pkgs.nodejs
    pkgs.nodePackages.typescript-language-server
    pkgs.yarn
    pkgs.replitPackages.prybar-python311
    pkgs.replitPackages.stderred
  ];
}
