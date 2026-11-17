"""
PWA (Progressive Web App) & Mobile App Launcher Integration Module
Enables 1-click installation on Android, iPhone/iPad (iOS), Windows PC, and macOS.
"""

import streamlit as st
import streamlit.components.v1 as components

def inject_pwa_headers():
    """
    Injects PWA metadata, manifest link, apple-touch-icon, and service worker registration into Streamlit HTML.
    """
    pwa_html = """
    <script>
    (function() {
        // Only inject once into the top-level document
        try {
            var doc = window.parent.document;
            if (!doc.getElementById('pwa-manifest-link')) {
                // Link Manifest
                var link = doc.createElement('link');
                link.id = 'pwa-manifest-link';
                link.rel = 'manifest';
                link.href = 'app/static/manifest.json';
                doc.head.appendChild(link);

                // Apple Touch Icon
                var appleIcon = doc.createElement('link');
                appleIcon.rel = 'apple-touch-icon';
                appleIcon.href = 'app/static/apple-touch-icon.png';
                doc.head.appendChild(appleIcon);

                // Apple Web App Capable Meta
                var metaCapable = doc.createElement('meta');
                metaCapable.name = 'apple-mobile-web-app-capable';
                metaCapable.content = 'yes';
                doc.head.appendChild(metaCapable);

                var metaStatus = doc.createElement('meta');
                metaStatus.name = 'apple-mobile-web-app-status-bar-style';
                metaStatus.content = 'black-translucent';
                doc.head.appendChild(metaStatus);

                var metaTitle = doc.createElement('meta');
                metaTitle.name = 'apple-mobile-web-app-title';
                metaTitle.content = 'F&O Screener';
                doc.head.appendChild(metaTitle);

                // Theme color
                var metaTheme = doc.createElement('meta');
                metaTheme.name = 'theme-color';
                metaTheme.content = '#00E676';
                doc.head.appendChild(metaTheme);

                // Service Worker Registration
                if ('serviceWorker' in window.parent.navigator) {
                    window.parent.navigator.serviceWorker.register('app/static/sw.js')
                        .catch(function(e) { /* ignore */ });
                }
            }
        } catch (e) {
            // Sandboxed iframe fallback
        }
    })();
    </script>
    """
    components.html(pwa_html, height=0, width=0)


def render_app_launcher_sidebar():
    """
    Renders an App Launcher and 1-Click Install guide in the sidebar.
    """
    with st.sidebar.expander("📲 App Launcher & 1-Click Setup", expanded=False):
        st.markdown(
            """
            <div style="font-size: 13px; line-height: 1.5; color: #E0E0E0;">
                <p><b>📱 Install on Mobile (Android / iPhone):</b></p>
                <ul style="padding-left: 18px; margin-top: -5px; margin-bottom: 8px;">
                    <li><b>Android:</b> Tap Chrome's <b>⋮</b> menu ➔ <i>"Install App"</i> or <i>"Add to Home Screen"</i>.</li>
                    <li><b>iPhone:</b> Tap Safari's <b>Share</b> button ➔ <i>"Add to Home Screen"</i>.</li>
                </ul>
                <p style="margin-top: 8px;"><b>💻 Install on PC / Mac:</b></p>
                <ul style="padding-left: 18px; margin-top: -5px; margin-bottom: 8px;">
                    <li><b>Chrome / Edge:</b> Click the <b>Install</b> icon (⊕ or 💻) on the right side of the address bar.</li>
                    <li><b>Mac:</b> Safari ➔ <i>File ➔ Add to Dock</i>.</li>
                </ul>
                <div style="background: rgba(0, 230, 118, 0.08); border-left: 3px solid #00E676; padding: 6px 10px; border-radius: 4px; margin-top: 8px; font-size: 12px; color: #B9F6CA;">
                    ✨ Opens fullscreen with custom trading icon without any browser URL bar clutter!
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
