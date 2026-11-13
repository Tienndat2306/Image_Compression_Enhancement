// static/js/charts.js
class CompressionChart {
    constructor(canvasId, data) {
        this.canvas = document.getElementById(canvasId);
        this.data = data;
        this.chart = null;
        this.init();
    }
    
    init() {
        this.createChart('doughnut');
    }
    
    createChart(type) {
        if (this.chart) {
            this.chart.destroy();
        }
        
        const ctx = this.canvas.getContext('2d');
        const config = this.getChartConfig(type);
        this.chart = new Chart(ctx, config);
    }
    
    getChartConfig(type) {
        const baseConfig = {
            data: {
                labels: ['Ảnh gốc', 'Ảnh đã nén'],
                datasets: [{
                    data: [this.data.original, this.data.compressed],
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(118, 75, 162, 0.8)'
                    ],
                    borderColor: [
                        'rgba(102, 126, 234, 1)',
                        'rgba(118, 75, 162, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { family: 'Inter', size: 12 }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const value = context.raw;
                                const total = this.data.original + this.data.compressed;
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${context.label}: ${value.toFixed(2)} KB (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        };
        
        if (type === 'bar') {
            return {
                type: 'bar',
                ...baseConfig,
                options: {
                    ...baseConfig.options,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Dung lượng (KB)',
                                font: { family: 'Inter' }
                            }
                        }
                    }
                }
            };
        }
        
        return {
            type: type,
            ...baseConfig
        };
    }
    
    updateData(newData) {
        this.data = newData;
        this.chart.data.datasets[0].data = [this.data.original, this.data.compressed];
        this.chart.update();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CompressionChart;
}