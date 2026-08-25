package ru.serverflow.cpubenchmark;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;

import com.termux.terminal.TerminalSession;
import com.termux.terminal.TerminalSessionClient;
import com.termux.view.TerminalView;
import com.termux.view.TerminalViewClient;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Minimal terminal shell around the native cpu_benchmark binary.
 *
 * The binary is shipped as jniLibs/&lt;abi&gt;/libcpu_benchmark.so so the OS
 * extracts it into the app's native-library directory at install time — the
 * only place under app-private storage where Android (10+) allows execve() of
 * a file the app itself shipped. A symlink named "cpu_benchmark" pointing at
 * it is created on PATH so the terminal reads like a normal shell.
 */
public class MainActivity extends Activity implements TerminalSessionClient, TerminalViewClient {

    private static final String LOG_TAG = "SFBenchTerminal";

    private TerminalView terminalView;
    private TerminalSession session;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        terminalView = new TerminalView(this, null);
        terminalView.setTextSize(32);
        terminalView.setTerminalViewClient(this);
        setContentView(terminalView);

        session = createSession();
        terminalView.attachSession(session);
        terminalView.requestFocus();
    }

    @Override
    protected void onResume() {
        super.onResume();
        terminalView.onScreenUpdated();
        showKeyboard();
    }

    private TerminalSession createSession() {
        String nativeLibDir = getApplicationInfo().nativeLibraryDir;
        String benchmarkBinary = nativeLibDir + "/libcpu_benchmark.so";

        File home = getFilesDir();
        home.mkdirs();
        String cwd = home.getAbsolutePath();

        // Symlink the real binary onto PATH under a friendly name (creating the
        // symlink itself, via "ln", is not subject to the noexec restriction —
        // only executing it afterwards is, and that target lives in the
        // permitted native-library directory).
        String startupCommand =
            "mkdir -p \"" + cwd + "/bin\" && " +
            "ln -sf \"" + benchmarkBinary + "\" \"" + cwd + "/bin/cpu_benchmark\"; " +
            "clear; " +
            "echo 'SFBench — Android CPU Benchmark'; " +
            "echo 'Повторный запуск: cpu_benchmark'; " +
            "echo; " +
            "cpu_benchmark; " +
            "exec /system/bin/sh -i";

        String[] args = { "-c", startupCommand };

        List<String> envList = new ArrayList<>();
        envList.add("HOME=" + cwd);
        envList.add("PATH=" + cwd + "/bin:/system/bin:/system/xbin");
        envList.add("TMPDIR=" + getCacheDir().getAbsolutePath());
        envList.add("TERM=xterm-256color");
        String[] env = envList.toArray(new String[0]);

        return new TerminalSession("/system/bin/sh", cwd, args, env, 2000, this);
    }

    private void showKeyboard() {
        InputMethodManager imm = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
        if (imm != null) {
            imm.showSoftInput(terminalView, InputMethodManager.SHOW_IMPLICIT);
        }
    }

    // ===== TerminalSessionClient =====

    @Override
    public void onTextChanged(TerminalSession changedSession) {
        terminalView.onScreenUpdated();
    }

    @Override
    public void onTitleChanged(TerminalSession changedSession) {
        // No title bar in this minimal shell.
    }

    @Override
    public void onSessionFinished(TerminalSession finishedSession) {
        finish();
    }

    @Override
    public void onCopyTextToClipboard(TerminalSession session, String text) {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null) {
            clipboard.setPrimaryClip(ClipData.newPlainText("", text));
        }
    }

    @Override
    public void onPasteTextFromClipboard(TerminalSession session) {
        ClipboardManager clipboard = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard != null && clipboard.hasPrimaryClip()) {
            CharSequence text = clipboard.getPrimaryClip().getItemAt(0).coerceToText(this);
            if (text != null) {
                session.getEmulator().paste(text.toString());
            }
        }
    }

    @Override
    public void onBell(TerminalSession session) {
        // Ignored.
    }

    @Override
    public void onColorsChanged(TerminalSession session) {
        terminalView.onScreenUpdated();
    }

    @Override
    public void onTerminalCursorStateChange(boolean state) {
    }

    @Override
    public Integer getTerminalCursorStyle() {
        return null;
    }

    // ===== TerminalViewClient =====

    @Override
    public float onScale(float scale) {
        return 1.0f;
    }

    @Override
    public void onSingleTapUp(MotionEvent e) {
        showKeyboard();
    }

    @Override
    public boolean shouldBackButtonBeMappedToEscape() {
        return false;
    }

    @Override
    public boolean shouldEnforceCharBasedInput() {
        return true;
    }

    @Override
    public boolean shouldUseCtrlSpaceWorkaround() {
        return false;
    }

    @Override
    public boolean isTerminalViewSelected() {
        return true;
    }

    @Override
    public void copyModeChanged(boolean copyMode) {
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent e, TerminalSession session) {
        return false;
    }

    @Override
    public boolean onKeyUp(int keyCode, KeyEvent e) {
        return false;
    }

    @Override
    public boolean onLongPress(MotionEvent event) {
        return false;
    }

    @Override
    public boolean readControlKey() {
        return false;
    }

    @Override
    public boolean readAltKey() {
        return false;
    }

    @Override
    public boolean readShiftKey() {
        return false;
    }

    @Override
    public boolean readFnKey() {
        return false;
    }

    @Override
    public boolean onCodePoint(int codePoint, boolean ctrlDown, TerminalSession session) {
        return false;
    }

    @Override
    public void onEmulatorSet() {
    }

    @Override
    public void logError(String tag, String message) {
        Log.e(tag, message);
    }

    @Override
    public void logWarn(String tag, String message) {
        Log.w(tag, message);
    }

    @Override
    public void logInfo(String tag, String message) {
        Log.i(tag, message);
    }

    @Override
    public void logDebug(String tag, String message) {
        Log.d(tag, message);
    }

    @Override
    public void logVerbose(String tag, String message) {
        Log.v(tag, message);
    }

    @Override
    public void logStackTraceWithMessage(String tag, String message, Exception e) {
        Log.e(tag, message, e);
    }

    @Override
    public void logStackTrace(String tag, Exception e) {
        Log.e(tag, "Error", e);
    }
}
