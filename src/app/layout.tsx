import type { Metadata, Viewport } from "next";
import "./simulator.css";

export const metadata: Metadata = {
  title: "Jevduck | Microduck Simulator",
  description: "Microduck simulation workspace with MuJoCo physics, original ONNX policies, live telemetry and Jev action control.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1, themeColor: "#0d1117" };

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
