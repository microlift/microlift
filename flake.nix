{
  description = "microlift - x86-64 to Cranelift IR lifter";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "clippy" "rust-analyzer" ];
        };

        crossGcc = pkgs.pkgsCross.gnu64.buildPackages.gcc;
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            pkgs.pkg-config
            # The `unicorn-oracle` feature builds Unicorn's vendored C core from
            # source via its `unicorn-engine-sys` build script (cmake + ninja).
            pkgs.cmake
            pkgs.ninja
          ];

          buildInputs = [
            rustToolchain
            pkgs.capstone  # required for --features disasm
            pkgs.unicorn   # optional: system libunicorn for --features unicorn-oracle
          ] ++ pkgs.lib.optionals (system != "x86_64-linux") [
            crossGcc
          ];

          RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
        };
      }
    );
}
