"""
PWA (Progressive Web App) & Mobile App Launcher Integration Module
Enables 1-click installation on Android, iPhone/iPad (iOS), Windows PC, and macOS.
"""

import streamlit as st
import streamlit.components.v1 as components

def inject_pwa_headers():
    """
    Injects PWA metadata, manifest link, apple-touch-icon, and service worker registration into Streamlit HTML.
    Also injects high-contrast persistent custom scrollbar CSS and mobile touch responsiveness directly into the top-level parent DOM.
    """
    pwa_html = """
    <script>
    (function() {
        try {
            var doc = window.parent.document;

            // Link PWA Manifest and Icons (once)
            if (!doc.getElementById('pwa-manifest-link')) {
                var link = doc.createElement('link');
                link.id = 'pwa-manifest-link';
                link.rel = 'manifest';
                link.href = 'app/static/manifest.json';
                doc.head.appendChild(link);

                var appleIcon = doc.createElement('link');
                appleIcon.rel = 'apple-touch-icon';
                appleIcon.href = 'app/static/apple-touch-icon.png';
                doc.head.appendChild(appleIcon);

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

                var metaTheme = doc.createElement('meta');
                metaTheme.name = 'theme-color';
                metaTheme.content = '#00E676';
                doc.head.appendChild(metaTheme);

                if ('serviceWorker' in window.parent.navigator) {
                    window.parent.navigator.serviceWorker.register('app/static/sw.js')
                        .catch(function(e) { /* ignore */ });
                }
            }

            // Unconditionally inject or update Custom High-Contrast Table Scrollbar & Mobile Responsiveness CSS
            var scrollStyle = doc.getElementById('custom-table-scrollbar-style');
            if (!scrollStyle) {
                scrollStyle = doc.createElement('style');
                scrollStyle.id = 'custom-table-scrollbar-style';
                doc.head.appendChild(scrollStyle);
            }
            scrollStyle.innerHTML = `
                /* Prevent Streamlit / Glide Data Grid containers from clipping scrollbars */
                [data-testid="stDataFrame"],
                .stDataFrame,
                [data-testid="stDataFrameResizable"],
                .stDataFrameGlideDataEditor {
                    overflow: visible !important;
                }

                /* WebKit scrollbars (disables macOS auto-hide overlay behavior) */
                ::-webkit-scrollbar {
                    -webkit-appearance: none !important;
                    width: 11px !important;
                    height: 11px !important;
                    display: block !important;
                }
                ::-webkit-scrollbar-track {
                    background: #0F172A !important;
                    border-radius: 5px !important;
                }
                ::-webkit-scrollbar-thumb {
                    background: rgba(59, 130, 246, 0.5) !important;
                    border-radius: 5px !important;
                    border: 1px solid rgba(15, 23, 42, 0.8) !important;
                    min-width: 25px !important;
                    min-height: 25px !important;
                }
                ::-webkit-scrollbar-thumb:hover {
                    background: #60A5FA !important;
                    box-shadow: 0 0 10px rgba(96, 165, 250, 0.9) !important;
                }
                ::-webkit-scrollbar-corner {
                    background: #0F172A !important;
                }

                /* High-contrast vibrant scrollbars inside all Streamlit tables */
                [data-testid="stDataFrame"] ::-webkit-scrollbar,
                .stDataFrame ::-webkit-scrollbar,
                .dvn-scroller::-webkit-scrollbar {
                    -webkit-appearance: none !important;
                    width: 11px !important;
                    height: 11px !important;
                    display: block !important;
                }
                [data-testid="stDataFrame"] ::-webkit-scrollbar-track,
                .stDataFrame ::-webkit-scrollbar-track,
                .dvn-scroller::-webkit-scrollbar-track {
                    background: #0B1120 !important;
                    border-radius: 5px !important;
                }
                [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb,
                .stDataFrame ::-webkit-scrollbar-thumb,
                .dvn-scroller::-webkit-scrollbar-thumb {
                    background: #3B82F6 !important;
                    border-radius: 5px !important;
                    border: 1.5px solid #0F172A !important;
                }
                [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb:hover,
                .stDataFrame ::-webkit-scrollbar-thumb:hover,
                .dvn-scroller::-webkit-scrollbar-thumb:hover {
                    background: #93C5FD !important;
                    box-shadow: 0 0 12px rgba(147, 197, 253, 0.95) !important;
                }

                /* Firefox standards */
                * {
                    scrollbar-color: #3B82F6 #0F172A !important;
                    scrollbar-width: thin !important;
                }

                /* ========================================================= */
                /* Mobile, iPad & Tablet Device Responsive Optimizations    */
                /* ========================================================= */
                @media (max-width: 768px) {
                    /* Compact scrollbars for smaller touch viewports */
                    ::-webkit-scrollbar,
                    [data-testid="stDataFrame"] ::-webkit-scrollbar,
                    .stDataFrame ::-webkit-scrollbar,
                    .dvn-scroller::-webkit-scrollbar {
                        width: 6px !important;
                        height: 6px !important;
                    }

                    /* Native smooth inertial momentum touch swiping */
                    .dvn-scroller {
                        -webkit-overflow-scrolling: touch !important;
                        touch-action: pan-x pan-y !important;
                    }

                    /* Prevent horizontal page overscroll while preserving table swiping */
                    body, html {
                        overflow-x: hidden !important;
                    }

                    /* Compact metric card spacing for mobile screens */
                    .metric-card {
                        padding: 8px 10px !important;
                        margin-bottom: 8px !important;
                    }
                    .metric-val {
                        font-size: 1.25rem !important;
                    }
                    .metric-lbl {
                        font-size: 0.70rem !important;
                    }
                }
            `;
        } catch (e) {
            // Sandboxed fallback
        }
    })();
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(pwa_html, height=1, width="stretch")
    else:
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
