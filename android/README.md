# SFBench Terminal (Android)

Тонкая обёртка вокруг нативного `cpu_benchmark` в виде Android-приложения:
окно терминала (библиотеки Termux `terminal-view` / `terminal-emulator`) и
обычная программная клавиатура — как в Termux, только без пакетного
менеджера и лишнего функционала.

## Как это устроено

- `MainActivity` поднимает `TerminalView`, запускает `/system/bin/sh` внутри
  псевдотерминала и один раз автоматически прогоняет `cpu_benchmark`, после
  чего оставляет обычный интерактивный шелл (можно запускать бенчмарк снова
  командой `cpu_benchmark`, или любые другие команды).
- Сам бинарник `cpu_benchmark` собирается нативно (CMake + Android NDK) и
  кладётся как `jniLibs/<abi>/libcpu_benchmark.so`. Это не настоящая
  разделяемая библиотека — так он просто оказывается в единственном месте
  внутри приватного хранилища приложения, откуда Android (10+) разрешает
  `execve()` собственных файлов (везде в `/data/data/<pkg>/...` смонтировано
  `noexec`). При старте на него создаётся симлинк `cpu_benchmark` в `$PATH`.

## Сборка

Полностью автоматизирована в CI: [`.github/workflows/android-apk.yml`](../.github/workflows/android-apk.yml)
на каждый push/PR, затрагивающий `android/`, `src/`, `cmake/` или
`CMakeLists.txt`, собирает нативный бинарник под `arm64-v8a` и `armeabi-v7a`,
затем — debug-APK. Готовый файл прикладывается к прогону workflow как
артефакт `sfbench-terminal-debug-apk`.

Локально (нужны Android SDK/NDK и JDK 17):

```bash
# 1. собрать нативный бинарник под нужный ABI, например arm64-v8a
cmake -B build-android -S .. \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_PORTABLE=ON -DBUILD_TESTS=OFF
cmake --build build-android --target cpu_benchmark -j

mkdir -p app/src/main/jniLibs/arm64-v8a
cp build-android/cpu_benchmark app/src/main/jniLibs/arm64-v8a/libcpu_benchmark.so

# 2. собрать APK
gradle assembleDebug   # или ./gradlew, если сгенерировать wrapper: gradle wrapper
```

## Известные ограничения

- Проект не содержит Gradle wrapper (`gradlew`) — CI ставит нужную версию
  Gradle сам через `gradle/actions/setup-gradle`. Для локальной сборки нужен
  установленный Gradle (или сгенерируйте wrapper командой `gradle wrapper`).
- APK не подписан релизным ключом — CI собирает только debug-сборку.
- Ни сама сборка, ни установка APK не проверялись на реальном устройстве —
  код написан по документированному API `termux-app` v0.118.3, но первый
  прогон CI может потребовать мелких правок.
