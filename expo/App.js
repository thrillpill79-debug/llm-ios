// PocketGPT wrapped for Expo Go.
//
// Expo Go can only run JavaScript — it cannot load llama.cpp's native code —
// so this hosts the web app in a WebView. That means the same WebKit engine,
// and the same memory ceiling, as Safari: roughly 1-2 GB, so ~1.7B parameter
// models. For 3B and larger you need the native app (see docs/TESTFLIGHT.md).
//
// What this does add: an app icon, a fullscreen UI with no browser chrome,
// and a free install through Expo Go.
import { useRef, useState } from "react";
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { StatusBar } from "expo-status-bar";
import { SafeAreaView } from "react-native-safe-area-context";
import { WebView } from "react-native-webview";

const APP_URL = "https://thrillpill79-debug.github.io/llm-ios/";

export default function App() {
  const webview = useRef(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(null);

  const reload = () => {
    setFailed(null);
    setLoading(true);
    webview.current?.reload();
  };

  return (
    <SafeAreaView style={styles.root} edges={["top", "left", "right"]}>
      <StatusBar style="light" />

      {failed ? (
        <View style={styles.center}>
          <Text style={styles.title}>Could not load PocketGPT</Text>
          <Text style={styles.detail}>{failed}</Text>
          <TouchableOpacity style={styles.button} onPress={reload}>
            <Text style={styles.buttonText}>Try again</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <WebView
          ref={webview}
          source={{ uri: APP_URL }}
          style={styles.web}
          // the model is stored by the page itself (origin-private filesystem),
          // so storage must persist between launches
          domStorageEnabled
          javaScriptEnabled
          cacheEnabled
          // a multi-hundred-megabyte download must not be interrupted
          mediaPlaybackRequiresUserAction={false}
          allowsInlineMediaPlayback
          originWhitelist={["*"]}
          setSupportMultipleWindows={false}
          // keep long-running WebAssembly work alive in the background
          allowsBackForwardNavigationGestures={false}
          onLoadEnd={() => setLoading(false)}
          onError={({ nativeEvent }) =>
            setFailed(nativeEvent.description ?? "Network error")
          }
          onHttpError={({ nativeEvent }) =>
            setFailed(`Server returned ${nativeEvent.statusCode}`)
          }
        />
      )}

      {loading && !failed && (
        <View style={styles.loader} pointerEvents="none">
          <ActivityIndicator size="large" color="#7c6cff" />
          <Text style={styles.loaderText}>Starting PocketGPT…</Text>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#101018" },
  web: { flex: 1, backgroundColor: "#101018" },
  loader: {
    ...StyleSheet.absoluteFillObject,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#101018",
    gap: 12,
  },
  loaderText: { color: "#8a8aa0", fontSize: 13 },
  center: { flex: 1, alignItems: "center", justifyContent: "center", padding: 28, gap: 12 },
  title: { color: "#e8e8f0", fontSize: 17, fontWeight: "600" },
  detail: { color: "#8a8aa0", fontSize: 13, textAlign: "center" },
  button: {
    backgroundColor: "#7c6cff",
    paddingHorizontal: 22,
    paddingVertical: 11,
    borderRadius: 12,
    marginTop: 8,
  },
  buttonText: { color: "#fff", fontWeight: "600" },
});
