document.addEventListener('DOMContentLoaded', function() {
    
    // --- [NEW] Tab Navigation Logic ---
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Get the target tab ID from the button's 'data-tab' attribute
            const targetTab = button.getAttribute('data-tab');

            // 1. Deactivate all buttons and content
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // 2. Activate the clicked button
            button.classList.add('active');
            
            // 3. Activate the corresponding content
            document.getElementById(targetTab).classList.add('active');
        });
    });
    // --- [END NEW] Tab Navigation Logic ---


    // --- Historical Chart Logic ---
    let regimeChart = null; // Global chart variable
    
    document.getElementById('run-historical').addEventListener('click', function() {
        const startDate = document.getElementById('start_date').value;
        const endDate = document.getElementById('end_date').value;

        fetch('/analyze_range', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ start_date: startDate, end_date: endDate }),
        })
        .then(response => response.json())
        .then(data => {
            renderRegimeChart(data);
        });
    });

    function renderRegimeChart(data) {
        const ctx = document.getElementById('regimeChart').getContext('2d');
        
        const chartData = {
            labels: data.map(d => d.date),
            datasets: [{
                label: 'S&P 500 Price',
                data: data.map(d => ({ x: d.date, y: d.price, regime: d.regime })),
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 2,
                pointRadius: 0,
                yAxisID: 'yPrice',
            }]
        };

        const regimeColors = {
            'Bull': 'rgba(40, 167, 69, 0.2)',
            'Bear': 'rgba(220, 53, 69, 0.2)',
            'Sideways': 'rgba(255, 193, 7, 0.2)'
        };

        if (regimeChart) {
            regimeChart.destroy();
        }

        regimeChart = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month'
                        }
                    },
                    yPrice: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'S&P 500 Price'
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y;
                                const regime = context.raw.regime;
                                if (regime) {
                                    label += ` (Regime: ${regime})`;
                                }
                                return label;
                            }
                        }
                    },
                }
            },
            plugins: [{
                id: 'regimeBackground',
                beforeDatasetsDraw(chart, args, options) {
                    const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
                    const segments = createRegimeSegments(data);

                    segments.forEach(segment => {
                        ctx.save();
                        ctx.fillStyle = regimeColors[segment.regime] || 'rgba(200, 200, 200, 0.2)';
                        const startPixel = x.getPixelForValue(new Date(segment.start));
                        const endPixel = x.getPixelForValue(new Date(segment.end));
                        ctx.fillRect(startPixel, top, endPixel - startPixel, bottom - top);
                        ctx.restore();
                    });
                }
            }]
        });
    }

    function createRegimeSegments(data) {
        if (data.length === 0) return [];
        
        let segments = [];
        let currentRegime = data[0].regime;
        let currentStart = data[0].date;

        for (let i = 1; i < data.length; i++) {
            if (data[i].regime !== currentRegime) {
                segments.push({ start: currentStart, end: data[i-1].date, regime: currentRegime });
                currentRegime = data[i].regime;
                currentStart = data[i].date;
            }
        }
        segments.push({ start: currentStart, end: data[data.length - 1].date, regime: currentRegime });
        return segments;
    }


    // --- "What-If" Simulator Logic ---
    document.getElementById('simulator-form').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const data = {
            log_return: document.getElementById('log_return').value,
            volatility: document.getElementById('volatility').value,
            momentum: document.getElementById('momentum').value,
            slope: document.getElementById('slope').value,
            yr5_change: document.getElementById('yr5_change').value,
            yr10_change: document.getElementById('yr10_change').value,
            yr20_change: document.getElementById('yr20_change').value,
            yr30_change: document.getElementById('yr30_change').value,
        };

        fetch('/simulate_scenario', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(result => {
            const resultDiv = document.getElementById('simulator-result');
            if (result.error) {
                resultDiv.className = 'regime-badge error';
                resultDiv.textContent = result.error;
            } else {
                resultDiv.className = `regime-badge ${result.regime_class}`;
                resultDiv.textContent = result.predicted_regime;
            }
        });
    });
});