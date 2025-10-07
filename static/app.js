// static/app.js

// --- Tab Navigation ---
function showTab(tabName) {
    document.querySelectorAll('.tab-content, .tab-button').forEach(el => el.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    event.currentTarget.classList.add('active');
}

// --- Charting Logic (Historical Range) ---
const rangeForm = document.getElementById('range-form');
const loader = document.getElementById('loader');
const ctx = document.getElementById('regimeChart').getContext('2d');
let regimeChart = null;

// --- UPDATED: New color palette for the white theme ---
const regimeColors = {
    'Bull': '#e9f5ec',     // Light Green
    'Bear': '#fdecea',     // Light Red
    'Sideways': '#fff8e1'  // Light Yellow
};

if (rangeForm) {
    rangeForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        loader.style.display = 'block';
        const startDate = document.getElementById('start_date').value;
        const endDate = document.getElementById('end_date').value;
        try {
            const response = await fetch('/analyze_range', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ start_date: startDate, end_date: endDate }),
            });
            const data = await response.json();
            renderChart(data);
        } catch (error) { console.error('Error fetching range data:', error); } 
        finally { loader.style.display = 'none'; }
    });
}

function renderChart(data) {
    if (regimeChart) { regimeChart.destroy(); }
    
    regimeChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'S&P 500 Close Price',
                data: data.map(d => d.price),
                borderColor: '#007bff', // Updated line color to match theme
                borderWidth: 2,
                tension: 0.1,
                pointRadius: 0,
            }]
        },
        options: {
            scales: {
                x: { type: 'time', time: { unit: 'month' }, ticks: { color: '#555' }, grid: { color: 'rgba(0, 0, 0, 0.05)' } },
                y: { ticks: { color: '#555' }, grid: { color: 'rgba(0, 0, 0, 0.1)' } }
            },
            plugins: { legend: { labels: { color: '#333' } } }
        },
        plugins: [{
            id: 'backgrounds',
            beforeDraw(chart, args, options) {
                const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
                ctx.save();
                data.forEach((item, index) => {
                    if (index === 0) return;
                    const prevItem = data[index - 1];
                    const startX = x.getPixelForValue(new Date(prevItem.date));
                    const endX = x.getPixelForValue(new Date(item.date));
                    ctx.fillStyle = regimeColors[prevItem.regime]; // Uses the new light colors
                    ctx.fillRect(startX, top, endX - startX, bottom - top);
                });
                ctx.restore();
            }
        }]
    });
}

// --- Scenario Simulator Logic ---
const scenarioForm = document.getElementById('scenario-form');
const scenarioResultCard = document.getElementById('scenario-result-card');
const scenarioResultDiv = document.getElementById('scenario-result');

if (scenarioForm) {
    scenarioForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const payload = {
            yr5_change: document.getElementById('yr5_change').value, yr10_change: document.getElementById('yr10_change').value,
            yr20_change: document.getElementById('yr20_change').value, yr30_change: document.getElementById('yr30_change').value,
            log_return: document.getElementById('log_return').value, volatility: document.getElementById('volatility').value,
        };
        try {
            const response = await fetch('/simulate_scenario', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const result = await response.json();
            if (result.error) { alert(result.error); } 
            else {
                scenarioResultDiv.textContent = result.predicted_regime;
                scenarioResultDiv.className = 'regime';
                scenarioResultDiv.classList.add(result.regime_class);
                scenarioResultCard.style.display = 'block';
            }
        } catch (error) { console.error('Error simulating scenario:', error); }
    });
}