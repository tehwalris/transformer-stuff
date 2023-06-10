{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  nativeBuildInputs = with pkgs; [
    gcc
    cudatoolkit
    gdb
    cudaPackages.nsight_compute
    hip
    rocthrust
    rocprim
  ];
  shellHook = ''
    export ROCM_INCLUDES="-isystem ${pkgs.rocthrust}/include"
  '';
}
