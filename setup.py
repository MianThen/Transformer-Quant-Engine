from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str = "") -> None:
        super().__init__(name, sources=[])
        self.source_dir = str(Path(source_dir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, extension: CMakeExtension) -> None:
        import pybind11

        output_dir = Path(self.get_ext_fullpath(extension.name)).resolve().parent
        config = "Debug" if self.debug else "Release"
        cmake = _cmake_executable()
        arguments = [
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir.as_posix()}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output_dir.as_posix()}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DQBT_BUILD_PYTHON=ON",
            "-DQBT_BUILD_LIVE_ENGINE=OFF",
            f"-DQBT_ENABLE_ML={_cmake_bool(os.getenv('QBT_ENABLE_ML', 'OFF'))}",
            f"-DQBT_ML_BACKEND={os.getenv('QBT_ML_BACKEND', 'onnxruntime')}",
        ]
        onnxruntime_root = os.getenv("ONNXRUNTIME_ROOT")
        if onnxruntime_root:
            arguments.append(f"-DONNXRUNTIME_ROOT={Path(onnxruntime_root).resolve()}")
        generator = os.getenv("CMAKE_GENERATOR")
        if generator:
            arguments.extend(["-G", generator])
        make_program = os.getenv("CMAKE_MAKE_PROGRAM")
        if make_program:
            arguments.append(f"-DCMAKE_MAKE_PROGRAM={make_program}")
        compiler = os.getenv("CXX")
        if compiler:
            arguments.append(f"-DCMAKE_CXX_COMPILER={compiler}")
        if sys.platform == "darwin":
            arguments.extend(_macos_cmake_arguments())
        configured_build_dir = os.getenv("QBT_CMAKE_BUILD_DIR")
        if configured_build_dir:
            build_dir = Path(configured_build_dir)
        elif os.name == "nt":
            build_dir = Path(tempfile.gettempdir()) / f"qbt-cmake-{sys.implementation.cache_tag}"
        else:
            build_dir = Path(self.build_temp) / extension.name
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [cmake, extension.source_dir, *arguments], cwd=build_dir, check=True
        )
        jobs = int(os.getenv("QBT_BUILD_JOBS", "2" if os.name == "nt" else str(os.cpu_count() or 2)))
        subprocess.run(
            [cmake, "--build", ".", "--target", "cpp_engine", "--config", config,
             "--parallel", str(jobs)],
            cwd=build_dir,
            check=True,
        )


def _cmake_executable() -> str:
    configured = os.getenv("CMAKE_BIN")
    if configured:
        return configured
    discovered = shutil.which("cmake")
    if discovered:
        return discovered
    macos_app = Path("/Applications/CMake.app/Contents/bin/cmake")
    if macos_app.is_file():
        return str(macos_app)
    raise RuntimeError("CMake executable was not found; set CMAKE_BIN")


def _cmake_bool(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"1", "ON", "TRUE", "YES"}:
        return "ON"
    if normalized in {"0", "OFF", "FALSE", "NO", ""}:
        return "OFF"
    raise RuntimeError("QBT_ENABLE_ML must be ON or OFF")


def _macos_cmake_arguments() -> list[str]:
    deployment_target = os.getenv("MACOSX_DEPLOYMENT_TARGET", "11.0")
    cflags = str(sysconfig.get_config_var("CFLAGS") or "")
    tokens = shlex.split(cflags)
    architectures = []
    for index, token in enumerate(tokens[:-1]):
        if token == "-arch" and tokens[index + 1] not in architectures:
            architectures.append(tokens[index + 1])
    arguments = [f"-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment_target}"]
    if architectures:
        arguments.append(f"-DCMAKE_OSX_ARCHITECTURES={';'.join(architectures)}")
    return arguments


setup(
    packages=[],
    ext_modules=[CMakeExtension("cpp_engine")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
