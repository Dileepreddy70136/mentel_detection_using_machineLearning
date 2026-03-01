/**
 * main.js — MindCare AI Dashboard
 * ─────────────────────────────────────────────────────────────
 * Fixes:
 *  • Real-time Chart.js updates from /api/history (SQLite)
 *  • History fetched from DB (not just last response)
 *  • .slice().reverse() instead of mutating .reverse()
 *  • Badge colour mapped from server badge field (success/warning/danger)
 *  • Confidence display
 *  • Score bar animated correctly
 *  • Stats loaded from /api/stats on page load
 */

document.addEventListener('DOMContentLoaded', async () => {

    // ── Date Header ─────────────────────────────────────────────
    const dateEl = document.getElementById('currentDate');
    if (dateEl) {
        const opts = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        dateEl.textContent = new Date().toLocaleDateString('en-US', opts);
    }

    // ── Chart.js Setup ──────────────────────────────────────────
    const ctxEl = document.getElementById('moodChart');
    let moodChart = null;

    if (ctxEl) {
        moodChart = new Chart(ctxEl.getContext('2d'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Wellness Score',
                    data: [],
                    borderColor: '#00e0ff',
                    backgroundColor: 'rgba(0, 224, 255, 0.07)',
                    tension: 0.45,
                    fill: true,
                    pointBackgroundColor: ctx => {
                        // Colour each dot by its score
                        const v = ctx.parsed?.y ?? 50;
                        return v >= 65 ? '#00e0ff' : v >= 30 ? '#ff9300' : '#ff0064';
                    },
                    pointBorderColor: '#0f172a',
                    pointRadius: 6,
                    pointHoverRadius: 9,
                }]
            },
            options: {
                responsive: true,
                animation: { duration: 600, easing: 'easeInOutQuart' },
                plugins: { legend: { display: false } },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8', callback: v => v + '%' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', maxTicksLimit: 8 }
                    }
                }
            }
        });
    }

    // ── DOM Refs ────────────────────────────────────────────────
    const analyzeBtn = document.getElementById('analyzeBtn');
    const moodInput = document.getElementById('moodInput');
    const resultCard = document.getElementById('resultCard');
    const loadingEl = document.getElementById('loading');
    const scoreValueEl = document.getElementById('scoreValue');
    const scoreBarEl = document.getElementById('scoreBar');
    const moodBadgeEl = document.getElementById('moodBadge');
    const moodBadgeSpan = moodBadgeEl?.querySelector('span');
    const confidenceEl = document.getElementById('confidenceValue');
    const suggestionList = document.getElementById('suggestionList');
    const historyTbody = document.querySelector('#historyTable tbody');
    const journalTextEl = document.getElementById('journalText');
    const journalCopyBtn = document.getElementById('journalCopyBtn');


    // Stats cards
    const statHappy = document.getElementById('statHappy');
    const statStressed = document.getElementById('statStressed');
    const statDepressed = document.getElementById('statDepressed');
    const statAvg = document.getElementById('statAvg');
    const statModelF1 = document.getElementById('statModelF1');

    // ── Badge Style Helper ──────────────────────────────────────
    const BADGE_CLASSES = ['badge-success', 'badge-warning', 'badge-danger'];

    function applyBadge(el, badge) {
        el.classList.remove(...BADGE_CLASSES);
        el.classList.add(`badge-${badge}`);
    }

    // ── Journal Copy Button ─────────────────────────────────────
    if (journalCopyBtn) {
        journalCopyBtn.addEventListener('click', () => {
            const text = journalTextEl?.textContent?.trim();
            if (!text) return;
            navigator.clipboard.writeText(text).then(() => {
                journalCopyBtn.textContent = 'Copied!';
                journalCopyBtn.classList.add('copied');
                setTimeout(() => {
                    journalCopyBtn.textContent = 'Copy';
                    journalCopyBtn.classList.remove('copied');
                }, 2000);
            }).catch(() => {
                journalCopyBtn.textContent = 'Failed';
                setTimeout(() => journalCopyBtn.textContent = 'Copy', 1500);
            });
        });
    }

    // ── Load History from SQLite (/api/history) ─────────────────
    async function loadHistory() {
        try {
            const res = await fetch('/api/history?limit=15');
            const data = await res.json();       // array, newest first

            renderHistoryTable(data);
            renderChart(data.slice().reverse()); // chart: oldest → newest
        } catch (err) {
            console.warn('Failed to load history:', err);
        }
    }

    // ── Load Stats ──────────────────────────────────────────────
    async function loadStats() {
        try {
            const res = await fetch('/api/stats');
            const data = await res.json();

            if (statHappy) statHappy.textContent = data.happy ?? '--';
            if (statStressed) statStressed.textContent = data.stressed ?? '--';
            if (statDepressed) statDepressed.textContent = data.depressed ?? '--';
            if (statAvg) statAvg.textContent = (data.avg_score ?? '--') + '%';
            if (statModelF1) statModelF1.textContent = (data.model_f1 ?? '--') + ' F1';
        } catch (err) {
            console.warn('Failed to load stats:', err);
        }
    }

    // ── Render Chart from SQLite History ───────────────────────
    function renderChart(entries) {
        if (!moodChart) return;

        // Use entries already in oldest-first order
        const labels = entries.map(e => e.timestamp.split(' ')[1].slice(0, 5)); // HH:MM
        const scores = entries.map(e => e.score);

        moodChart.data.labels = labels;
        moodChart.data.datasets[0].data = scores;
        moodChart.update('active');
    }

    // ── Render History Table ────────────────────────────────────
    function renderHistoryTable(entries) {
        if (!historyTbody) return;
        historyTbody.innerHTML = '';

        entries.forEach(entry => {
            const badge = entry.badge || 'success';
            const time = (entry.timestamp || '').split(' ')[1] || '--:--';
            const conf = entry.confidence
                ? (entry.confidence * 100).toFixed(0) + '%'
                : '--';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${time}</td>
                <td title="${entry.text_snippet}">${entry.text_snippet}</td>
                <td><span class="badge badge-${badge}">${entry.prediction}</span></td>
                <td>${entry.score}%</td>
                <td style="color:var(--text-dim);font-size:0.82rem">${conf}</td>
            `;
            historyTbody.appendChild(tr);
        });
    }

    // ── ANALYZE Button ──────────────────────────────────────────
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {
            const text = moodInput?.value.trim() ?? '';
            if (!text) { alert('Please share your thoughts first!'); return; }

            // Loading state — Fix 2: disable + visual feedback prevents spam clicks
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:8px"></div> Analyzing...';
            loadingEl?.classList.remove('hidden');
            resultCard?.classList.add('hidden');

            try {
                const res = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });

                const data = await res.json();

                if (!res.ok) {
                    alert(`Error: ${data.error || 'Server error'}`);
                    return;
                }

                // ── Show Result Card ────────────────────────────
                resultCard?.classList.remove('hidden');

                // Badge (with CSS colour transition)
                if (moodBadgeSpan) moodBadgeSpan.textContent = data.prediction;
                if (moodBadgeEl) applyBadge(moodBadgeEl, data.badge);

                // Score bar
                if (scoreValueEl) scoreValueEl.textContent = `${data.score}%`;
                if (scoreBarEl) {
                    scoreBarEl.style.width = '0%';
                    // Trigger CSS transition
                    requestAnimationFrame(() => {
                        scoreBarEl.style.width = `${data.score}%`;
                    });
                }

                // Confidence
                if (confidenceEl) confidenceEl.textContent = `${data.confidence}%`;

                // Suggestions
                if (suggestionList) {
                    suggestionList.innerHTML = '';
                    data.suggestions.forEach(s => {
                        const li = document.createElement('li');
                        li.textContent = s;
                        suggestionList.appendChild(li);
                    });
                }

                // ── Journal Entry (fade in) ─────────────────────
                if (journalTextEl && data.journal) {
                    journalTextEl.classList.remove('visible');
                    journalTextEl.textContent = data.journal;
                    // Slight delay so the fade-in feels deliberate
                    setTimeout(() => journalTextEl.classList.add('visible'), 120);
                }

                // ── Refresh Graph + Table from DB ───────────────
                // Small delay so the DB write completes first
                await new Promise(r => setTimeout(r, 150));
                await loadHistory();
                await loadStats();

            } catch (err) {
                console.error('Fetch error:', err);
                alert('Connection failed. Is Flask running on port 5000?');
            } finally {
                analyzeBtn.disabled = false;
                analyzeBtn.innerHTML = 'Analyze Text <i class="fas fa-magic"></i>';
                loadingEl?.classList.add('hidden');
            }
        });
    }

    // ── Webcam Section ──────────────────────────────────────────
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const webcamFeed = document.getElementById('webcamFeed');
    const faceOverlay = document.getElementById('faceOverlay');
    const faceStatusSpan = document.querySelector('#faceStatus span');
    let isWebcamOn = false;
    let webcamStream = null;
    let faceInterval = null;

    if (startWebcamBtn) {
        startWebcamBtn.addEventListener('click', async () => {
            if (!isWebcamOn) {
                try {
                    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    if (webcamFeed) webcamFeed.srcObject = webcamStream;
                    faceOverlay?.classList.add('hidden');
                    if (faceStatusSpan) faceStatusSpan.textContent = 'Scanning…';
                    isWebcamOn = true;
                    runMockFaceAnalysis();
                } catch {
                    alert('Camera access denied. Allow camera in browser settings.');
                }
            } else {
                stopWebcam();
            }
        });
    }

    function stopWebcam() {
        webcamStream?.getTracks().forEach(t => t.stop());
        if (webcamFeed) webcamFeed.srcObject = null;
        faceOverlay?.classList.remove('hidden');
        if (faceStatusSpan) faceStatusSpan.textContent = 'Offline';
        clearInterval(faceInterval);
        isWebcamOn = false;
    }

    function runMockFaceAnalysis() {
        const emotions = [
            ['😐 Neutral', '#94a3b8'],
            ['😊 Happy', '#00e0ff'],
            ['😢 Sad', '#7000ff'],
            ['😰 Stressed', '#ff9300'],
            ['🧐 Focused', '#00e0ff'],
            ['😠 Angry', '#ff0064'],
        ];
        faceInterval = setInterval(() => {
            if (!isWebcamOn) { clearInterval(faceInterval); return; }
            const [label, color] = emotions[Math.floor(Math.random() * emotions.length)];
            if (faceStatusSpan) {
                faceStatusSpan.textContent = label;
                faceStatusSpan.style.color = color;
            }
        }, 2500);
    }

    // ── Initial Data Load ───────────────────────────────────────
    await loadHistory();
    await loadStats();
});
