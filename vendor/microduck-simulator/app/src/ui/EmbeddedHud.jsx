import { bootLog, useGame } from "../store.js";

// The host app provides commands; this small overlay reports the real
// simulator's loading/error state and measured loop rates.
export default function EmbeddedHud() {
  const ready = useGame((s) => s.bootDone);
  const failed = useGame((s) => s.bootFailed);
  const mode = useGame((s) => s.modeLabel);
  const telemetry = useGame((s) => s.telemetry);
  const error = failed ? bootLog.findLast((e) => e.label.startsWith(">> "))?.label.slice(3) : null;
  return <div style={{ position: "fixed", inset: 0, zIndex: 10, pointerEvents: "none", fontFamily: "ui-monospace, monospace", color: "#e8e9df", fontSize: 11 }}>
    {(!ready || failed) && <div role={failed ? "alert" : "status"} style={{ position: "absolute", inset: 0, display: "grid", placeContent: "center", gap: 14, textAlign: "center", padding: 24, background: "#08080ce8" }}>
      <strong style={{ fontSize: 15 }}>{failed ? "The simulator could not start" : "Starting the real Microduck simulator…"}</strong>
      <span style={{ maxWidth: 420, lineHeight: 1.7, color: "#b5bbb0" }}>{failed ? error || "Reload to try again." : "Loading the robot model and trained movement policies."}</span>
    </div>}
    {ready && <div style={{ position: "absolute", left: 14, bottom: 12, display: "flex", flexWrap: "wrap", gap: 12, padding: "7px 10px", borderRadius: 6, background: "#08080ca8", color: "#bcc6b4" }} aria-label="Measured simulator telemetry">
      <span>{mode}</span><span>{telemetry.fps} fps</span><span>{telemetry.ctrlHz} Hz control</span>
    </div>}
  </div>;
}
