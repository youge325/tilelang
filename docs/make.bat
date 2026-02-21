@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
    echo.installed, then set the SPHINXBUILD environment variable to point
    echo.to the full path of the 'sphinx-build' executable. Alternatively you
    echo.may add the Sphinx directory to PATH.
    echo.
    echo.If you don't have Sphinx installed, grab it from
    echo.https://www.sphinx-doc.org/
    exit /b 1
)

if "%1" == "" goto help
if "%1" == "html-en" goto html-en
if "%1" == "html-zh" goto html-zh
if "%1" == "html-all" goto html-all

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html-en
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR%\en %SPHINXOPTS% %O%
goto end

:html-zh
%SPHINXBUILD% -M html %SOURCEDIR%\zh_CN %BUILDDIR%\zh_CN %SPHINXOPTS% %O%
goto end

:html-all
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR%\en %SPHINXOPTS% %O%
%SPHINXBUILD% -M html %SOURCEDIR%\zh_CN %BUILDDIR%\zh_CN %SPHINXOPTS% %O%
echo ^<!DOCTYPE html^>^<html^>^<head^>^<meta http-equiv="refresh" content="0;url=en/html/index.html"^>^</head^>^<body^>^<p^>^<a href="en/html/index.html"^>English^</a^> ^| ^<a href="zh_CN/html/index.html"^>中文^</a^>^</p^>^</body^>^</html^> > %BUILDDIR%\index.html
echo Build complete for EN and ZH_CN.
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
