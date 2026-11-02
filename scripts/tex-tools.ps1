# Shared by report build entry points. Prefer TEX_BIN, otherwise use PATH.
function Get-FoodLensTexTool {
    param([Parameter(Mandatory = $true)][string]$Name)
    if ($env:TEX_BIN) {
        $suffix = if ($env:OS -eq 'Windows_NT') { '.exe' } else { '' }
        $candidate = Join-Path $env:TEX_BIN ($Name + $suffix)
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Cannot find $Name in TEX_BIN: $env:TEX_BIN"
        }
        return $candidate
    }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) {
        throw "Cannot find $Name. Install TeX Live and add its bin directory to PATH, or set TEX_BIN."
    }
    return $command.Source
}
