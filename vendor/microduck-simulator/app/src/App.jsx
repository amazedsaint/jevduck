// App shell: layers in z order - scene canvas (1), halftone (2), CRT (5),
// HUD + touch overlay (9-12), BIOS readout (20), title menu (30). Also owns
// the pre-game device detection (gamepad, touch) and the title -> enter
// flow. The title menu carries its own loading gate (single central
// spinner), so no separate preboot veil.
import { useEffect } from "react";
import GameCanvas from "./scene/GameCanvas.jsx";
import { Halftone, CrtOverlay } from "./ui/Overlays.jsx";
import Hud from "./ui/Hud.jsx";
import TouchOverlay from "./ui/TouchOverlay.jsx";
import BiosOverlay from "./ui/BiosOverlay.jsx";
import TitleMenu from "./ui/TitleMenu.jsx";
import { useGame } from "./store.js";
import { EMBEDDED, mountEmbeddedBridge } from "./game/embedded.js";

export default function App() {
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const unmountBridge = mountEmbeddedBridge();

    // Gamepad presence for the tutorial variant, before the game controller
    // takes over polling (the game loop keeps mirroring it afterwards).
    const pollPad = () => {
      const gp = [...(navigator.getGamepads?.() ?? [])].find((p) => p && p.connected);
      useGame.setState({ padConnected: !!gp });
    };
    window.addEventListener("gamepadconnected", pollPad);
    window.addEventListener("gamepaddisconnected", pollPad);
    pollPad();

    // Touch detection, latching: (pointer: coarse) OR any real touch proves
    // the device (DevTools device modes and hybrid laptops often fail the
    // media query). __microduckTouched hands the proof to controls/touch.js
    // which boots later. ?touch=1 forces the thumbs on any device.
    const coarse = matchMedia("(pointer: coarse)");
    const armTouch = () => {
      window.__microduckTouched = true;
      useGame.setState({ touchMode: true });
    };
    const onCoarse = () => coarse.matches && armTouch();
    if (coarse.matches) armTouch();
    else coarse.addEventListener("change", onCoarse);
    window.addEventListener("touchstart", armTouch, { once: true, passive: true });
    if (params.get("touch") === "1") armTouch();

    // ?move=<ref>: request a community move (Academy D7). ref is an
    // "org/repo" Hub move, "session:<id>[:<round>]" (a private Academy run,
    // served same-origin) or a direct .onnx URL. ?policy= is the legacy
    // alias, and ?session=<id>[&round=<r>] is sugar for move=session:<id>:<r>.
    // Only the request is recorded here; the game loads it after boot
    // (never blocking it) and flips the status to active/error itself.
    let moveRef = params.get("move") || params.get("policy");
    if (!moveRef && params.get("session")) {
      const round = params.get("round");
      moveRef = `session:${params.get("session")}${round ? `:${round}` : ""}`;
    }
    if (moveRef && !EMBEDDED) {
      useGame.setState({
        customPolicy: { ref: moveRef, name: moveRef, status: "loading" },
      });
    }

    // The title menu opens right away and holds its own loading gate
    // (central spinner) until logo, fonts, duck stage and game boot are
    // in. ?boot=1 skips the title entirely - testing hook.
    if (EMBEDDED || params.get("boot") === "1") {
      useGame.setState({ prebootDone: true, entered: true, menuOpen: false });
    } else {
      useGame.setState({ prebootDone: true, menuOpen: true });
    }

    return () => {
      unmountBridge();
      window.removeEventListener("gamepadconnected", pollPad);
      window.removeEventListener("gamepaddisconnected", pollPad);
      coarse.removeEventListener("change", onCoarse);
      window.removeEventListener("touchstart", armTouch);
    };
  }, []);

  return (
    <>
      <GameCanvas />
      {!EMBEDDED && <>
        <Halftone />
        <CrtOverlay />
        <Hud />
        <TouchOverlay />
        <BiosOverlay />
        <TitleMenu />
      </>}
    </>
  );
}
