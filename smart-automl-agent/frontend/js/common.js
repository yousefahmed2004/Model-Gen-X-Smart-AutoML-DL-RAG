/* ============================================================
   Model Gen X — shared frontend runtime
   - API client (fetch wrapper with auth)
   - Theme toggle (persisted)
   - i18n (en / ar) with RTL
   - Toast notifications
   ============================================================ */

// ------------------- Config -------------------
const API_BASE = (() => {
    const host = window.location.hostname;
    // Production server — always use HTTPS domain
    if (host === 'modelgenx.site' || host === 'www.modelgenx.site') {
        return 'https://modelgenx.site';
    }
    const meta = document.querySelector('meta[name="api-base"]');
    if (meta && meta.content) return meta.content;
    if (host === 'localhost' || host === '127.0.0.1' || host === '[::1]') {
        return 'http://localhost:8000';
    }
    return 'https://modelgenx.site';
})();
window.API_BASE = API_BASE;

// Store this page's origin so the OAuth callback HTML page can redirect back here.
// Works for both http://localhost:5500 and file:// origins.
(function storeOrigin() {
    const origin = location.origin === 'null'
        // file:// — store the directory path so we can build absolute paths
        ? location.href.replace(/\/[^/]*$/, '').replace(/\/pages$/, '')
        : location.origin;
    localStorage.setItem('saa.frontend_origin', origin);
})();

// ------------------- Auth token -------------------
const TOKEN_KEY = 'saa.token';
const USER_KEY = 'saa.user';

const auth = {
    get token() { return localStorage.getItem(TOKEN_KEY); },
    set token(v) { v ? localStorage.setItem(TOKEN_KEY, v) : localStorage.removeItem(TOKEN_KEY); },
    get user() {
        const raw = localStorage.getItem(USER_KEY);
        try { return raw ? JSON.parse(raw) : null; } catch { return null; }
    },
    set user(v) { v ? localStorage.setItem(USER_KEY, JSON.stringify(v)) : localStorage.removeItem(USER_KEY); },
    clear() { this.token = null; this.user = null; },
    requireAuth(redirect = 'login.html') {
        if (!this.token) {
            window.location.href = redirect;
            return false;
        }
        return true;
    },
};
window.auth = auth;

// Pick up token from OAuth redirect fragment: #access_token=...
(function readFragmentToken() {
    if (!location.hash.startsWith('#access_token=')) return;
    const t = location.hash.replace('#access_token=', '');
    auth.token = t;
    history.replaceState({}, '', location.pathname + location.search);
})();

// ------------------- API client -------------------
async function api(path, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (auth.token) headers['Authorization'] = `Bearer ${auth.token}`;
    let body = opts.body;
    if (body && !(body instanceof FormData) && typeof body !== 'string') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify(body);
    }
    const res = await fetch(`${API_BASE}${path}`, { ...opts, headers, body });
    if (res.status === 401) {
        auth.clear();
        if (!location.pathname.endsWith('login.html')) {
            location.href = 'login.html'  // already correct for pages/;
        }
        throw new Error('Unauthorized');
    }
    if (!res.ok) {
        let detail = `${res.status} ${res.statusText}`;
        try {
            const errBody = await res.json();
            if (errBody.detail) detail = typeof errBody.detail === 'string' ? errBody.detail : JSON.stringify(errBody.detail);
        } catch { }
        throw new Error(detail);
    }
    if (res.status === 204) return null;
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? res.json() : res.blob();
}
window.api = api;

// ------------------- Toasts -------------------
function ensureToastHost() {
    let host = document.querySelector('.toast-host');
    if (!host) {
        host = document.createElement('div');
        host.className = 'toast-host';
        document.body.appendChild(host);
    }
    return host;
}

function toast(message, type = '') {
    const host = ensureToastHost();
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    host.appendChild(el);
    setTimeout(() => {
        el.style.transition = 'opacity 0.4s, transform 0.4s';
        el.style.opacity = '0';
        el.style.transform = 'translateX(20px)';
        setTimeout(() => el.remove(), 400);
    }, 3800);
}
window.toast = toast;

// ------------------- Theme -------------------
const THEME_KEY = 'saa.theme';
function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
    const ico = document.querySelector('[data-theme-icon]');
    if (ico) ico.textContent = theme === 'dark' ? '☾' : '☼';
}
function initTheme() {
    const saved = localStorage.getItem(THEME_KEY) || 'dark';
    applyTheme(saved);
    document.querySelectorAll('[data-theme-toggle]').forEach(btn => {
        btn.addEventListener('click', () => {
            const cur = document.documentElement.getAttribute('data-theme');
            applyTheme(cur === 'dark' ? 'light' : 'dark');
        });
    });
}
window.applyTheme = applyTheme;

// ------------------- i18n -------------------
const STRINGS = {
    en: {
        'nav.features': 'Features',
        'nav.pricing': 'Pricing',
        'nav.docs': 'Docs',
        'nav.signin': 'Sign in',
        'nav.dashboard': 'Dashboard',
        'nav.logout': 'Logout',
        'cta.get_started': 'Get started',
        'cta.try_free': 'Try it free',
        'hero.eyebrow': 'AutoML, with words.',
        'hero.title_1': 'Build models',
        'hero.title_2': 'by describing',
        'hero.title_3': 'what you need.',
        'hero.sub': 'Upload a dataset. Tell the agent your goal. Get a trained, evaluated, deployable model in minutes — no notebooks, no boilerplate.',
        'common.loading': 'Loading',
        'common.train': 'Train',
        'common.predict': 'Predict',
        'common.upload': 'Upload',
    },
    ar: {
        'nav.features': 'المميزات',
        'nav.pricing': 'الأسعار',
        'nav.docs': 'الوثائق',
        'nav.signin': 'تسجيل الدخول',
        'nav.dashboard': 'لوحة التحكم',
        'nav.logout': 'خروج',
        'cta.get_started': 'ابدأ الآن',
        'cta.try_free': 'جرّب مجاناً',
        'hero.eyebrow': 'تعلم آلي، بالكلمات.',
        'hero.title_1': 'ابنِ نماذج',
        'hero.title_2': 'بوصف ما تحتاج',
        'hero.title_3': 'فقط لا أكثر.',
        'hero.sub': 'ارفع مجموعة بيانات. أخبر الوكيل بهدفك. واحصل على نموذج مُدرَّب وجاهز للنشر خلال دقائق — بدون أكواد.',
        'common.loading': 'جارِ التحميل',
        'common.train': 'تدريب',
        'common.predict': 'تنبؤ',
        'common.upload': 'رفع',
    },
};

const LANG_KEY = 'saa.lang';
function applyLang(lang) {
    const dict = STRINGS[lang] || STRINGS.en;
    document.documentElement.setAttribute('lang', lang);
    document.documentElement.setAttribute('dir', lang === 'ar' ? 'rtl' : 'ltr');
    localStorage.setItem(LANG_KEY, lang);
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (dict[key]) el.textContent = dict[key];
    });
    document.querySelectorAll('[data-i18n-attr]').forEach(el => {
        const [attr, key] = el.getAttribute('data-i18n-attr').split(':');
        if (dict[key]) el.setAttribute(attr, dict[key]);
    });
    const label = document.querySelector('[data-lang-label]');
    if (label) label.textContent = lang.toUpperCase();
}
function initLang() {
    const saved = localStorage.getItem(LANG_KEY) || 'en';
    applyLang(saved);
    document.querySelectorAll('[data-lang-toggle]').forEach(btn => {
        btn.addEventListener('click', () => {
            const cur = document.documentElement.getAttribute('lang') || 'en';
            applyLang(cur === 'en' ? 'ar' : 'en');
        });
    });
}
window.applyLang = applyLang;

// ------------------- Init on load -------------------
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initLang();
});