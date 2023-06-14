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
    cargo
    rustc
    rust-analyzer
    rustfmt
    llvmPackages.libclang
  ];
  shellHook = ''
    export ROCTHRUST_PATH="${pkgs.rocthrust}"
    export CUDA_NATIVE_PATH="${pkgs.cudatoolkit}"
    export NIX_CFLAGS_COMPILE="$(echo $NIX_CFLAGS_COMPILE | perl -pe 's/\s+-isystem ([^ ]+-(cudatoolkit|rocthrust)-[^ ]+)//g')"
    export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
  '';
}
