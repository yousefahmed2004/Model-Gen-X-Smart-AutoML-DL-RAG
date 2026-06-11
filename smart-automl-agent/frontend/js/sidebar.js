/* ============================================================
   App-shell shared helpers — sidebar, user load
   Model Gen X
   ============================================================ */

function iconSVG(name) {
    const map = {
        home: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 11l9-8 9 8M5 9.5V21h14V9.5"/></svg>',
        chat: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M21 11.5a8.4 8.4 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.4 8.4 0 0 1-3.8-.9L3 21l1.9-5.7a8.4 8.4 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.4 8.4 0 0 1 3.8-.9h.5a8.5 8.5 0 0 1 8 8v.5z"/></svg>',
        upload: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>',
        train: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 12h20M5 7l2 5-2 5M19 7l-2 5 2 5M12 3v18"/></svg>',
        results: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 3v18h18M7 16l4-4 3 3 5-7"/></svg>',
        rag: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/><path d="M8 10h8M8 14h5"/></svg>',
        play: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5 3v18l15-9z"/></svg>',
        price: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z"/><circle cx="7.5" cy="7.5" r="1.5"/></svg>',
        logout: '<svg class="side-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>',
    };
    return map[name] || '';
}

function renderSidebar(activeKey) {
    const links = [
        { key: 'dashboard', href: 'dashboard.html', label: 'Dashboard', icon: 'home' },
        { key: 'chat', href: 'chat.html', label: 'AI chat', icon: 'chat' },
        { key: 'upload', href: 'upload.html', label: 'Upload', icon: 'upload' },
        { key: 'training', href: 'training.html', label: 'Training', icon: 'train' },
        { key: 'results', href: 'results.html', label: 'Results', icon: 'results' },
        { key: 'playground', href: 'playground.html', label: 'Playground', icon: 'play' },
        { key: 'rag', href: 'rag_bots.html', label: 'RAG Bots', icon: 'rag' },
    ];
    const account = [
        { key: 'pricing', href: 'pricing.html', label: 'Pricing', icon: 'price' },
    ];

    const linkHTML = l => `
        <a class="side-link ${l.key === activeKey ? 'active' : ''}" href="${l.href}">
            ${iconSVG(l.icon)}<span>${l.label}</span>
        </a>`;

    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) return;

    // Read current theme/lang state
    const curTheme = localStorage.getItem('saa.theme') || 'dark';
    const curLang = localStorage.getItem('saa.lang') || 'en';

    sidebar.innerHTML = `
        <a class="brand" href="dashboard.html">
            <span class="brand-mark"></span>
            <strong>Model Gen X</strong>
        </a>
        <div class="side-section">
            <span class="side-heading">Workspace</span>
            ${links.map(linkHTML).join('')}
        </div>
        <div class="side-section">
            <span class="side-heading">Account</span>
            ${account.map(linkHTML).join('')}
        </div>
        <div class="side-footer">
            <div class="user-chip" id="userChip">
                <div class="user-avatar" id="userAvatar">·</div>
                <div class="user-info">
                    <div class="user-name" id="userName">—</div>
                    <div class="user-tokens"><span class="accent" id="userTokens">0</span> tokens</div>
                </div>
                <button class="btn-icon" id="logoutBtn" title="Sign out" style="border:none;width:32px;height:32px;">
                    ${iconSVG('logout')}
                </button>
            </div>
            <div class="row" style="gap:6px;margin-top:10px;justify-content:center;">
                <button
                    class="btn-icon"
                    id="sideThemeBtn"
                    title="Toggle theme"
                    style="width:34px;height:34px;font-size:15px;">
                    ${curTheme === 'dark' ? '☾' : '☼'}
                </button>
                <button
                    class="btn-icon"
                    id="sideLangBtn"
                    title="Language"
                    style="width:34px;height:34px;font-size:11px;font-family:'JetBrains Mono',monospace;">
                    ${curLang.toUpperCase()}
                </button>
            </div>
        </div>
    `;

    // Add hamburger button if not exists
    if (!document.getElementById('hamburgerBtn')) {
        const ham = document.createElement('button');
        ham.id = 'hamburgerBtn';
        ham.className = 'hamburger';
        ham.setAttribute('aria-label', 'Menu');
        ham.innerHTML = '<span></span><span></span><span></span>';
        document.body.appendChild(ham);
    }

    // Add overlay if not exists
    if (!document.getElementById('sidebarOverlay')) {
        const overlay = document.createElement('div');
        overlay.id = 'sidebarOverlay';
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
    }

    const ham = document.getElementById('hamburgerBtn');
    const overlay = document.getElementById('sidebarOverlay');

    function openSidebar() {
        sidebar.classList.add('open');
        overlay.classList.add('open');
    }
    function closeSidebar() {
        sidebar.classList.remove('open');
        overlay.classList.remove('open');
    }
    function toggleSidebar() {
        if (sidebar.classList.contains('open')) {
            closeSidebar();
        } else {
            openSidebar();
        }
    }

    ham.onclick = toggleSidebar;
    overlay.onclick = closeSidebar;

    // Close sidebar when a link is clicked on mobile
    sidebar.querySelectorAll('.side-link').forEach(l => {
        l.addEventListener('click', () => {
            if (window.innerWidth <= 640) closeSidebar();
        });
    });

    // Logout
    document.getElementById('logoutBtn')?.addEventListener('click', () => {
        auth.clear();
        location.href = 'login.html';
    });

    // Theme toggle — direct call, no data-attribute dependency
    document.getElementById('sideThemeBtn')?.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = cur === 'dark' ? 'light' : 'dark';
        // Update icon immediately
        document.getElementById('sideThemeBtn').textContent = next === 'dark' ? '☾' : '☼';
        if (window.applyTheme) {
            applyTheme(next);
        } else {
            document.documentElement.setAttribute('data-theme', next);
            localStorage.setItem('saa.theme', next);
        }
    });

    // Language toggle — direct call
    document.getElementById('sideLangBtn')?.addEventListener('click', () => {
        const cur = document.documentElement.getAttribute('lang') || 'en';
        const next = cur === 'en' ? 'ar' : 'en';
        // Update label immediately
        document.getElementById('sideLangBtn').textContent = next.toUpperCase();
        if (window.applyLang) {
            applyLang(next);
        } else {
            document.documentElement.setAttribute('lang', next);
            document.documentElement.setAttribute('dir', next === 'ar' ? 'rtl' : 'ltr');
            localStorage.setItem('saa.lang', next);
        }
    });
}

async function loadCurrentUser() {
    try {
        const user = await api('/api/auth/me');
        auth.user = user;

        const initials = (user.name || user.email || '?')
            .split(/\s+/).map(p => p[0]).slice(0, 2).join('').toUpperCase();

        const av = document.getElementById('userAvatar');
        if (av) {
            if (user.picture) {
                av.innerHTML = `<img src="${user.picture}" alt=""
                    style="width:100%;height:100%;border-radius:50%;object-fit:cover;" />`;
            } else {
                av.textContent = initials;
            }
        }

        const nm = document.getElementById('userName');
        if (nm) nm.textContent = user.name || user.email;

        const tk = document.getElementById('userTokens');
        if (tk) tk.textContent = user.tokens.toLocaleString();

        return user;
    } catch {
        // 401 handled by api() — redirects to login
    }
}

window.renderSidebar = renderSidebar;
window.loadCurrentUser = loadCurrentUser;