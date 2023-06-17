let
  fenix_overlay = import "${builtins.fetchTarball "https://github.com/nix-community/fenix/archive/main.tar.gz"}/overlay.nix";
  pkgs = import <nixpkgs> { overlays = [ fenix_overlay ]; };
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
      gcc
      gdb
      cudaPackages.cudatoolkit.out
      cudaPackages.cudatoolkit.lib
      cudaPackages.nsight_compute
      hip
      rocthrust
      rocprim
       (fenix.complete.withComponents [
        "cargo"
        "clippy"
        "rust-src"
        "rustc"
        "rustfmt"
      ])
      rust-analyzer-nightly
      llvmPackages.libclang
      rocm-device-libs
      ninja
    ];
    shellHook = ''
      export ROCTHRUST_PATH="${pkgs.rocthrust}"
      export CUDA_NATIVE_PATH="${pkgs.cudatoolkit}"
      export NIX_CFLAGS_COMPILE="$(echo $NIX_CFLAGS_COMPILE | perl -pe 's/\s+-isystem ([^ ]+-(cudatoolkit|rocthrust)-[^ ]+)//g')"
      export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
      export LD_LIBRARY_PATH=/run/opengl-driver/lib
      export ROCM_PATH="${pkgs.hip}"
      export HIP_PATH="${pkgs.hip}"
      export HSA_PATH="${pkgs.rocm-runtime}"
      export DEVICE_LIB_PATH="${pkgs.rocm-device-libs}/amdgcn/bitcode"
    '';
  }
