{
      description = "Env for GPT2";
      outputs = { self, nixpkgs }:
      let
        system = "x86_64-linux";
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };
      in {
        devShells.${system}.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cudatoolkit linuxPackages.nvidia_x11
            cudaPackages.cudnn
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5 stdenv.cc binutils
            ffmpeg
            python311
            python311Packages.pip
            python311Packages.numpy
            python311Packages.pytorch-bin
            python311Packages.virtualenv
          ];

          shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib"
          '';          
        };
      };
    }
