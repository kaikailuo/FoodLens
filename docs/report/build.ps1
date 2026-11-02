$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    . (Join-Path $PSScriptRoot '../../scripts/tex-tools.ps1')
    $texTools = @{ xelatex = (Get-FoodLensTexTool 'xelatex'); bibtex = (Get-FoodLensTexTool 'bibtex') }
    $buildDirectory = Join-Path $PSScriptRoot 'build/latex'
    New-Item -ItemType Directory -Path $buildDirectory -Force | Out-Null
    foreach ($step in @('xelatex', 'bibtex', 'xelatex', 'xelatex')) {
        if ($step -eq 'bibtex') {
            & ($texTools['bibtex']) 'build/latex/research_proposal'
        } else {
            & ($texTools['xelatex']) -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$buildDirectory" research_proposal.tex
        }
        if ($LASTEXITCODE -ne 0) { throw "$step failed (exit $LASTEXITCODE)" }
    }
    Copy-Item -LiteralPath (Join-Path $buildDirectory 'research_proposal.pdf') -Destination (Join-Path $PSScriptRoot 'research_proposal.pdf') -Force
    Write-Host 'Built research_proposal.pdf; intermediate files are in build/latex/.'
} finally {
    Pop-Location
}
