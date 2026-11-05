[CmdletBinding()]
param(
    [string]$Checkpoint,
    [string]$Data,
    [string[]]$CaseIds = @("bunny-0003"),
    [int[]]$Frames = @(19),
    [ValidateScript({ $_ -gt 0 })][int]$Repeats = 1,
    [ValidateSet("cpu", "cuda", "auto")][string]$Device = "cpu",
    [string]$BeforeCommit = "61b5119c006e73cb00139a1d54ee730aeb57df75",
    [string]$AfterCommit = "1c21310cb60c7203c877ffe48d343c2b39f23785"
)

function Assert-NativeSuccess([string]$Operation) {
    if ($LASTEXITCODE -ne 0) { throw "$Operation failed with exit code $LASTEXITCODE" }
}

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$previousPythonPath = $env:PYTHONPATH
$previousErrorPreference = $ErrorActionPreference
Push-Location -LiteralPath $projectRoot
try {
    $ErrorActionPreference = "Stop"
    if (-not $CaseIds.Count -or @($CaseIds | Select-Object -Unique).Count -ne $CaseIds.Count) { throw "CaseIds must be nonempty and unique" }
    if (-not $Frames.Count -or @($Frames | Where-Object { $_ -lt 0 }).Count -or @($Frames | Select-Object -Unique).Count -ne $Frames.Count) { throw "Frames must be unique nonnegative indices" }
    if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
        $latest = Get-ChildItem -Path "runs/bunny-fast-*/best.pt" -File -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($null -eq $latest) { throw "No Bunny best.pt found; provide -Checkpoint" }
        $Checkpoint = $latest.FullName
    }
    if ([string]::IsNullOrWhiteSpace($Data)) { $Data = "data/bunny-native-1000/collection.json" }
    $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).ProviderPath
    $dataPath = (Resolve-Path -LiteralPath $Data).ProviderPath
    $checkpointHash = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $dataHash = if (Test-Path -LiteralPath $dataPath -PathType Leaf) { (Get-FileHash -LiteralPath $dataPath -Algorithm SHA256).Hash }
    $beforeRef = (& git rev-parse --verify "$($BeforeCommit)^{commit}")
    Assert-NativeSuccess "Resolving the previous commit; fetch repository history if missing"
    $afterRef = (& git rev-parse --verify "$($AfterCommit)^{commit}")
    Assert-NativeSuccess "Resolving the updated commit; update the repository if missing"
    $tag = (Get-Date -Format "yyyyMMdd-HHmmss") + "-" + [guid]::NewGuid().ToString("N").Substring(0, 8)
    $runRoot = Join-Path $projectRoot "runs/corrector-compare-$tag"
    $sources = @{ before = "$runRoot/before-source"; after = "$runRoot/after-source" }
    New-Item -ItemType Directory -Path $runRoot | Out-Null
    & git archive --format=zip "--output=$runRoot/source.zip" $afterRef src
    Assert-NativeSuccess "Exporting the updated source"
    foreach ($destination in $sources.Values) { Expand-Archive -LiteralPath "$runRoot/source.zip" -DestinationPath $destination }
    $numericFiles = @("src/fracture_world/dynamics_runtime.py", "src/fracture_world/dynamics_corrector.py",
                      "src/fracture_world/dynamics_energy_solvers.py", "src/fracture_world/dynamics_quadratic_projection.py")
    & git archive --format=zip "--output=$runRoot/numeric-before.zip" $beforeRef @numericFiles
    Assert-NativeSuccess "Exporting the previous numerical corrector"
    Expand-Archive -LiteralPath "$runRoot/numeric-before.zip" -DestinationPath $sources.before -Force
    $adapter = @'


_baseline_correct = _correct

def _correct(*args, reuse_state=None, **kwargs):
    return _baseline_correct(*args, **kwargs)
'@
    [System.IO.File]::AppendAllText("$($sources.before)/src/fracture_world/dynamics_runtime.py", $adapter, [System.Text.UTF8Encoding]::new($false))
    Copy-Item -LiteralPath "configs/bunny-rollout.json" -Destination "$runRoot/rollout.json"
    $commonArguments = @("--checkpoint", $checkpointPath, "--data", $dataPath, "--specification", "$runRoot/rollout.json",
                         "--device", $Device, "--split", "test", "--case-ids") + $CaseIds + @("--frames") + $Frames
    Write-Host "Checkpoint: $checkpointPath"
    Write-Host "Dataset: $dataPath"
    Write-Host "Output: $runRoot"
    $rows = [System.Collections.Generic.List[object]]::new()
    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
        $reports = @{}
        $order = if ($repeat % 2) { @("before", "after") } else { @("after", "before") }
        foreach ($mode in $order) {
            $env:PYTHONPATH = "$($sources[$mode])/src"
            $loadedPackage = (& python -B -c "import fracture_world; print(fracture_world.__file__)")
            Assert-NativeSuccess "Checking the imported $mode package"
            if ([System.IO.Path]::GetFullPath($loadedPackage) -ne [System.IO.Path]::GetFullPath("$($sources[$mode])/src/fracture_world/__init__.py")) { throw "Wrong $mode source package imported" }
            $output = "$runRoot/$mode-$repeat"
            $modeArguments = $commonArguments + @("--output", $output)
            if ($mode -eq "before") { $modeArguments += "--cold-corrector" }
            Write-Host "Running $mode, repeat $repeat/$Repeats"
            try {
                $ErrorActionPreference = "Continue"
                & python -B -u -m fracture_world dynamics-matched @modeArguments 2>&1 | Tee-Object -FilePath "$output.log"
            }
            finally { $ErrorActionPreference = "Stop" }
            Assert-NativeSuccess "$mode comparison"
            $report = Read-Json "$output/matched.json"
            if ($report.status -ne "completed") { throw "$mode comparison was not completed" }
            $reports[$mode] = @{ directory = $output; report = $report }
        }
        $before = $reports.before.report
        $after = $reports.after.report
        if ($before.checkpoint_sha256 -ne $checkpointHash -or $after.checkpoint_sha256 -ne $checkpointHash) { throw "Checkpoint changed between runs" }
        if (($before.configuration | ConvertTo-Json -Depth 10 -Compress) -ne ($after.configuration | ConvertTo-Json -Depth 10 -Compress)) { throw "Physical configurations differ" }
        foreach ($case in $before.cases) {
            $matchingCase = @($after.cases | Where-Object { $_.case_id -eq $case.case_id })
            if ($matchingCase.Count -ne 1) { throw "Matching after case missing" }
            $beforeCase = Read-Json "$($reports.before.directory)/$($case.report)"
            $afterCase = Read-Json "$($reports.after.directory)/$($matchingCase[0].report)"
            foreach ($step in $beforeCase.frames) {
                $matches = @($afterCase.frames | Where-Object { $_.frame -eq $step.frame })
                if ($matches.Count -ne 1) { throw "Matching after frame missing" }
                $next = $matches[0]
                if ($step.status -ne "completed" -or $next.status -ne "completed") { throw "A transition was not evaluated completely" }
                if ($step.input_hash -ne $next.input_hash -or ($step.candidate_pool -join ",") -ne ($next.candidate_pool -join ",")) { throw "Inputs or fracture candidate pools differ" }
                foreach ($old in $step.candidates | Where-Object { $_.method -in @("central", "consensus") }) {
                    $matches = @($next.candidates | Where-Object { $_.method -eq $old.method -and $_.candidate_hash -eq $old.candidate_hash })
                    if ($matches.Count -ne 1) { throw "Matching numerical candidate missing" }
                    $new = $matches[0]
                    if ($old.proposal_hash -ne $new.proposal_hash) { throw "Neural proposals differ; the comparison is not controlled" }
                    $paired = [bool]($old.accepted -and $new.accepted)
                    $oldEvaluation = if ($old.PSObject.Properties["evaluation"]) { $old.evaluation } else { $null }
                    $newEvaluation = if ($new.PSObject.Properties["evaluation"]) { $new.evaluation } else { $null }
                    $rows.Add([pscustomobject]@{
                        Repeat = $repeat; Case = $case.case_id; Frame = $step.frame; Method = $old.method
                        CandidateHash = $old.candidate_hash; PairedCompleted = $paired
                        BeforeStatus = $old.status; AfterStatus = $new.status
                        BeforeAccepted = $old.accepted; AfterAccepted = $new.accepted
                        BeforeVerified = $old.verified_constraints; AfterVerified = $new.verified_constraints
                        BeforeSelected = ($step.methods.($old.method).selected_candidate -eq $old.candidate)
                        AfterSelected = ($next.methods.($new.method).selected_candidate -eq $new.candidate)
                        BeforeSeconds = $old.corrector_runtime_seconds; AfterSeconds = $new.corrector_runtime_seconds
                        Speedup = $(if ($paired -and $new.corrector_runtime_seconds -gt 0) { $old.corrector_runtime_seconds / $new.corrector_runtime_seconds } else { $null })
                        BeforeIterations = $old.iterations; AfterIterations = $new.iterations
                        BeforePositionRMSE = $(if ($oldEvaluation) { $oldEvaluation.position_mass_weighted_rmse_m }); AfterPositionRMSE = $(if ($newEvaluation) { $newEvaluation.position_mass_weighted_rmse_m })
                        BeforeVelocityRMSE = $(if ($oldEvaluation) { $oldEvaluation.velocity_mass_weighted_rmse_m_per_s }); AfterVelocityRMSE = $(if ($newEvaluation) { $newEvaluation.velocity_mass_weighted_rmse_m_per_s })
                        BeforeFractureF1 = $(if ($oldEvaluation) { $oldEvaluation.new_fracture.f1 }); AfterFractureF1 = $(if ($newEvaluation) { $newEvaluation.new_fracture.f1 })
                    })
                }
            }
        }
    }
    if ((Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $checkpointHash) { throw "Checkpoint changed during comparison" }
    if ($dataHash -and (Get-FileHash -LiteralPath $dataPath -Algorithm SHA256).Hash -ne $dataHash) { throw "Dataset collection changed during comparison" }
    $rows | Export-Csv -LiteralPath "$runRoot/comparison.csv" -NoTypeInformation -Encoding utf8
    $summary = [ordered]@{
        BeforeCommit = $beforeRef; AfterCommit = $afterRef; Checkpoint = $checkpointPath; CheckpointSHA256 = $checkpointHash
        Data = $dataPath; DataSHA256 = $dataHash; Device = $Device; Repeats = $Repeats; CaseIds = $CaseIds; Frames = $Frames
        ConfigurationSHA256 = (Get-FileHash -LiteralPath "$runRoot/rollout.json" -Algorithm SHA256).Hash
        UpdatedSourceSHA256 = (Get-FileHash -LiteralPath "$runRoot/source.zip" -Algorithm SHA256).Hash
        PreviousNumericalSourceSHA256 = (Get-FileHash -LiteralPath "$runRoot/numeric-before.zip" -Algorithm SHA256).Hash
        Protocol = "Shared updated frontend, predictor and evaluator; previous versus updated numerical core on identical frozen one-step inputs"
        Timing = "Per-candidate numerical corrector wall time; excludes neural proposal generation and subsequent independent evaluation"
        Rows = @($rows.ToArray())
    }
    $summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath "$runRoot/comparison.json" -Encoding utf8
    $rows | Select-Object Repeat, Case, Frame, Method, BeforeStatus, AfterStatus, BeforeSeconds, AfterSeconds | Format-Table -AutoSize
    Write-Host "Results: $runRoot/comparison.json"
    Write-Host "CSV: $runRoot/comparison.csv"
}
finally {
    $env:PYTHONPATH = $previousPythonPath
    $ErrorActionPreference = $previousErrorPreference
    Pop-Location
}
