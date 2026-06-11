/* Landing page interactions */

// ---------- Terminal typing animation ----------
(function () {
    const typed = document.getElementById('termTyped');
    const reply = document.getElementById('termReply');
    const metrics = document.getElementById('termMetrics');
    if (!typed) return;

    const userMessage = 'predict iris species from sepal & petal measurements';
    const replySteps = [
        '✓ dataset profiled — 150 rows × 5 columns',
        '✓ task detected: classification (3 classes)',
        '✓ training 4 candidates with 3-fold CV…',
        '✓ best: Random Forest (cv-accuracy 0.946)',
        '✓ saved to ~/trained_models/iris.joblib',
    ];

    let idx = 0;
    function typeChar() {
        if (idx >= userMessage.length) {
            setTimeout(runReply, 500);
            return;
        }
        typed.textContent += userMessage[idx++];
        setTimeout(typeChar, 35 + Math.random() * 50);
    }

    let stepIdx = 0;
    function runReply() {
        if (stepIdx >= replySteps.length) {
            metrics.style.display = 'grid';
            return;
        }
        reply.textContent += (stepIdx > 0 ? '\n' : '') + replySteps[stepIdx++];
        setTimeout(runReply, 600);
    }

    setTimeout(typeChar, 1500);
})();

// ---------- Stat counter on intersection ----------
(function () {
    const stats = document.querySelectorAll('[data-count]');
    if (!stats.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const el = entry.target;
            const target = parseInt(el.dataset.count, 10);
            const prefix = el.querySelector('.prefix')?.textContent || '';
            const suffix = el.querySelector('.suffix')?.textContent || '';
            let cur = 0;
            const step = Math.max(1, Math.ceil(target / 60));
            const tick = () => {
                cur = Math.min(target, cur + step);
                el.innerHTML = `${prefix ? `<span class="prefix">${prefix}</span>` : ''}${cur}${suffix ? `<span class="suffix">${suffix}</span>` : ''}`;
                if (cur < target) requestAnimationFrame(tick);
            };
            tick();
            observer.unobserve(el);
        });
    }, { threshold: 0.4 });

    stats.forEach(s => observer.observe(s));
})();

// ---------- Scroll-trigger fade-in for sections ----------
(function () {
    const els = document.querySelectorAll('.feature, .workflow-step, .testimonial, .faq-item');
    if (!els.length) return;
    els.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(24px)';
        el.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
    });
    const io = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.style.opacity = '1';
                e.target.style.transform = 'translateY(0)';
                io.unobserve(e.target);
            }
        });
    }, { threshold: 0.15 });
    els.forEach(el => io.observe(el));
})();
