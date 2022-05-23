How to get the code
===

```
git clone https://partner-code.googlesource.com/deepmind-androidenv-vokram

```
When cloning for the first time, it will ask you for username and password. Don't enter anything, just press enter with empty fields. It will give you a link with a command to setup your gitcookies. After you ran this command, repeat the clone command above.

How to push modifications
===

```
git add <filename>
git commit -m "<commit message>"
git push origin HEAD:refs/for/master
```

How to build
===

```sh
ANDROID_SDK_ROOT=~/Android/Sdk/ ./gradlew build
```
The output of the build is in `./app/build/outputs/apk/debug/app-debug.apk`
