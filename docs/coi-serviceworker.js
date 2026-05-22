/*!
 * coi-serviceworker — Cross-Origin Isolation polyfill.
 *
 * Why this file exists:
 *   GitHub Pages cannot set custom HTTP headers (COOP/COEP), which are
 *   required for `SharedArrayBuffer` (needed by multi-threaded ffmpeg.wasm).
 *   This service worker intercepts responses and adds those headers on the
 *   client side, enabling SharedArrayBuffer without server-side config.
 *
 *   The LyricFuse MVP currently uses single-threaded ffmpeg core which does
 *   NOT require SharedArrayBuffer — so this is a no-op in the current build,
 *   but it's preloaded so we can swap to @ffmpeg/core-mt later (for a ~2-4x
 *   speed boost on subtitle burning) with zero infrastructure changes.
 *
 * Upstream: https://github.com/gzuidhof/coi-serviceworker
 * License:  MIT — Guido Zuidhof and contributors
 */
let coepCredentialless = false;
if (typeof window === "undefined") {
  // ----- Service-worker context -----
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

  self.addEventListener("message", (ev) => {
    if (!ev.data) return;
    if (ev.data.type === "deregister") {
      self.registration
        .unregister()
        .then(() => self.clients.matchAll())
        .then((clients) => clients.forEach((client) => client.navigate(client.url)));
    } else if (ev.data.type === "coepCredentialless") {
      coepCredentialless = ev.data.value;
    }
  });

  self.addEventListener("fetch", (event) => {
    const r = event.request;
    if (r.cache === "only-if-cached" && r.mode !== "same-origin") return;

    const request =
      coepCredentialless && r.mode === "no-cors"
        ? new Request(r, { credentials: "omit" })
        : r;

    event.respondWith(
      fetch(request)
        .then((response) => {
          if (response.status === 0) return response;
          const newHeaders = new Headers(response.headers);
          newHeaders.set(
            "Cross-Origin-Embedder-Policy",
            coepCredentialless ? "credentialless" : "require-corp"
          );
          if (!coepCredentialless) {
            newHeaders.set("Cross-Origin-Resource-Policy", "cross-origin");
          }
          newHeaders.set("Cross-Origin-Opener-Policy", "same-origin");
          return new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: newHeaders,
          });
        })
        .catch((e) => console.error(e))
    );
  });
} else {
  // ----- Page context: register the service worker if needed -----
  (() => {
    const reloadedBySelf = window.sessionStorage.getItem("coiReloadedBySelf");
    window.sessionStorage.removeItem("coiReloadedBySelf");
    const coepDegrading = reloadedBySelf === "coepdegrade";

    const coi = {
      shouldRegister: () => !reloadedBySelf,
      shouldDeregister: () => false,
      coepCredentialless: () => true,
      coepDegrade: () => true,
      doReload: () => window.location.reload(),
      quiet: false,
      ...window.coi,
    };

    const n = navigator;

    if (n.serviceWorker && n.serviceWorker.controller) {
      n.serviceWorker.controller.postMessage({
        type: "coepCredentialless",
        value:
          coepDegrading || !coi.coepCredentialless()
            ? false
            : coi.coepCredentialless(),
      });
      if (coi.shouldDeregister()) {
        n.serviceWorker.controller.postMessage({ type: "deregister" });
      }
    }

    if (window.crossOriginIsolated !== false || !coi.shouldRegister()) return;

    if (!window.isSecureContext) {
      !coi.quiet &&
        console.log("COOP/COEP Service Worker not registered: secure context required.");
      return;
    }

    if (!n.serviceWorker) {
      !coi.quiet &&
        console.error("COOP/COEP Service Worker not registered: navigator.serviceWorker unavailable.");
      return;
    }

    n.serviceWorker.register(window.document.currentScript.src).then(
      (registration) => {
        !coi.quiet && console.log("COOP/COEP Service Worker registered", registration.scope);

        registration.addEventListener("updatefound", () => {
          !coi.quiet && console.log("Reloading to use updated COOP/COEP Service Worker.");
          window.sessionStorage.setItem("coiReloadedBySelf", "updatefound");
          coi.doReload();
        });

        if (registration.active && !n.serviceWorker.controller) {
          !coi.quiet && console.log("Reloading to use COOP/COEP Service Worker.");
          window.sessionStorage.setItem("coiReloadedBySelf", "active");
          coi.doReload();
        }
      },
      (err) => {
        !coi.quiet && console.error("COOP/COEP Service Worker registration failed:", err);
        if (coi.coepDegrade()) {
          window.sessionStorage.setItem("coiReloadedBySelf", "coepdegrade");
          coi.doReload();
        }
      }
    );
  })();
}
