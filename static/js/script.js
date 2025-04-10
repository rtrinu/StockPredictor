document.addEventListener('DOMContentLoaded', function() {
    const stockPlotSection = document.getElementById('stockPlotSection');
    const generatedStockPlot = document.getElementById('generatedStockPlot');
    const stockPlotCompanies = document.getElementById('stockPlotCompanies');
    const generatePlotBtn = document.getElementById('generatePlotBtn');

    // Generate Stock Plot on Page Load
    generateStockPlot();

    // Generate Plot Button Event Listener
    generatePlotBtn.addEventListener('click', generateStockPlot);

    function generateStockPlot() {
        fetch('/generate-stock-plot')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                // Update plot image
                generatedStockPlot.src = `/static/${data.filename}`;
                stockPlotCompanies.textContent = `Comparing ${data.companies[0]} and ${data.companies[1]}`;
                
                // Show stock plot section
                stockPlotSection.style.display = 'block';
            })
            .catch(error => {
                console.error('Error generating plot:', error);
                stockPlotCompanies.textContent = 'Failed to generate plot';
                stockPlotSection.style.display = 'block';
            });
    }
});