"""
VoiceTriage — Universal Web App Launcher (PC, Phone & Online)
=============================================================
Usage:
    python run.py               (Local PC + Phone on same Wi-Fi)
    python run.py --share       (Generates public secure HTTPS link for anywhere)
"""

import sys
import os
import socket
import argparse
import subprocess
import webbrowser
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

def get_local_ip() -> str:
    """Get the primary local IPv4 address (e.g. Wi-Fi IP)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def check_flask():
    """Verify core dependencies."""
    try:
        import flask        # noqa
        import flask_cors   # noqa
        return True
    except ImportError:
        return False

def install_flask():
    """Auto-install Flask if missing."""
    print("  Installing web server dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "flask", "flask-cors", "--quiet"])

def start_tunnel(port: int):
    """
    Attempt to create a free public HTTPS tunnel so phones anywhere in the
    world can access the app with working microphone permissions.
    """
    # Try pyngrok if installed
    try:
        from pyngrok import ngrok
        public_url = ngrok.connect(port).public_url
        if public_url.startswith("http://"):
            public_url = public_url.replace("http://", "https://")
        return public_url
    except Exception:
        pass

    # Try cloudflared if installed
    try:
        p = subprocess.Popen(["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(2.5)
        return "Check cloudflared terminal output for https://*.trycloudflare.com link"
    except Exception:
        pass

    return None

def main():
    parser = argparse.ArgumentParser(description="VoiceTriage Web Launcher")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default 5000)")
    parser.add_argument("--share", action="store_true", help="Create public secure HTTPS link for phones")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser on PC")
    args = parser.parse_args()

    port = args.port
    local_ip = get_local_ip()

    print()
    print("=" * 66)
    print("       VoiceTriage — AI Voice Symptom Triage Assistant")
    print("              Multi-Device Web Launcher (PC & Phone)")
    print("=" * 66)

    if not check_flask():
        install_flask()

    sys.path.insert(0, str(PROJECT_ROOT))
    from webapp.app import app

    pc_url = f"http://127.0.0.1:{port}"
    phone_wifi_url = f"http://{local_ip}:{port}"

    print(f"\n  [OK] Server is starting on all network interfaces:")
    print(f"  -------------------------------------------------------------")
    print(f"  * Local PC Access  :  {pc_url}")
    print(f"  * Phone (Wi-Fi)    :  {phone_wifi_url}")
    print(f"  -------------------------------------------------------------")

    public_tunnel_url = None
    if args.share:
        print("\n  [i] Starting secure public HTTPS tunnel for mobile phones...")
        try:
            # Check or install pyngrok if user requested --share
            try:
                import pyngrok
            except ImportError:
                print("  Installing pyngrok for public link...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok", "--quiet"])
            public_tunnel_url = start_tunnel(port)
            if public_tunnel_url:
                print(f"  * Public Online URL:  {public_tunnel_url}")
                print(f"    (Open this on ANY phone or PC anywhere in the world!)")
        except Exception as e:
            print(f"  [!] Note: To enable public URL, run: pip install pyngrok (Error: {e})")

    print("\n  Tips for Phone Access:")
    print("    1. PC & Phone on SAME Wi-Fi -> Open Phone (Wi-Fi) link above.")
    print("    2. For direct Voice/Mic on mobile browsers (Chrome/Safari):")
    print("       Run with '--share' flag for an HTTPS link:")
    print("       python run.py --share")
    print("    3. Or use Chrome's unsafely-treat-insecure-origin flag for local testing.")
    print("\n  Press Ctrl+C to stop.\n")

    # Auto-open browser on PC
    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(pc_url)
        import threading
        threading.Thread(target=open_browser, daemon=True).start()

    # Bind to 0.0.0.0 to listen on both 127.0.0.1 (PC) and local Wi-Fi IP (Phone)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
