#requires -Version 7.0
<#
.SYNOPSIS
Updates the dedicated vast1 SSH block and uploads the local bootstrap helper.
.EXAMPLE
./scripts/local/Prepare-Vast.ps1 'ssh -p 51729 root@66.115.179.150 -L 8080:localhost:8080' -WhatIf
.EXAMPLE
./scripts/local/Prepare-Vast.ps1 -HostName 66.115.179.150 -Port 51729 -User root
.NOTES
Does not run BOOTSTRAP.sh, authenticate Codex, rent a host, or start GPU work.
SSH options other than -p/-l/-L/-i are rejected; forwards and identity overrides
in a pasted command are intentionally not applied. The configured key is used.
#>
[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium', DefaultParameterSetName = 'Command')]
param(
    [Parameter(Position = 0, ParameterSetName = 'Command')][Alias('SshCommand')][string]$Connection,
    [Parameter(ParameterSetName = 'Endpoint')][string]$HostName,
    [Parameter(ParameterSetName = 'Endpoint')][int]$Port,
    [Parameter(ParameterSetName = 'Endpoint')][string]$User,
    [string]$ConfigPath = (Join-Path $HOME '.ssh/config'),
    [string]$IdentityFile = (Join-Path $HOME '.ssh/id_ed25519_vast_datn'),
    [string]$BootstrapPath = (Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) 'BOOTSTRAP.sh'),
    [switch]$SkipUpload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Test-VastEndpoint {
    param([string]$Name, [int]$SshPort, [string]$Login)
    if ([string]::IsNullOrWhiteSpace($Name)) { throw 'Supply the SSH hostname or IP address.' }
    if ($SshPort -lt 1 -or $SshPort -gt 65535) { throw 'Port must be between 1 and 65535.' }
    if ($Login -notmatch '^[A-Za-z_][A-Za-z0-9_.-]*[$]?$') { throw 'Invalid SSH user name.' }
    if ($Name.StartsWith('[') -and $Name.EndsWith(']')) { $Name = $Name.Substring(1, $Name.Length - 2) }
    $address = $null
    if ([System.Net.IPAddress]::TryParse($Name, [ref]$address)) {
        if ($Name.Contains('%')) { throw 'IPv6 zone identifiers are not supported; use a routable address.' }
    } elseif ($Name.Length -gt 253 -or $Name -notmatch '^(?=.{1,253}$)[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$' -or
        @($Name.Split('.') | Where-Object { $_.Length -gt 63 -or $_ -notmatch '^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$' }).Count -gt 0) {
        throw 'Invalid SSH hostname or IP address.'
    }
    [pscustomobject]@{ HostName = $Name; Port = $SshPort; User = $Login }
}

function ConvertFrom-VastSshCommand {
    param([string]$Text)
    # Tokenize data only. Never execute or expand a pasted command.
    if ([string]::IsNullOrWhiteSpace($Text) -or $Text -match '[\r\n`;$|&<>]') {
        throw 'Paste one plain SSH command without shell operators or line breaks.'
    }
    $tokens = [System.Collections.Generic.List[string]]::new()
    $pattern = '"([^"\r\n]*)"|''([^''\r\n]*)''|([^\s"'']+)'
    $matches = [regex]::Matches($Text, $pattern)
    $position = 0
    foreach ($match in $matches) {
        if ($Text.Substring($position, $match.Index - $position).Trim()) { throw 'Unmatched quote in SSH command.' }
        $tokens.Add(($match.Groups[1..3] | Where-Object Success | Select-Object -First 1).Value)
        $position = $match.Index + $match.Length
    }
    if ($Text.Substring($position).Trim() -or $tokens.Count -lt 2 -or $tokens[0] -notin @('ssh', 'ssh.exe')) {
        throw "Expected: ssh -p PORT USER@HOST (with spaces around options)."
    }
    $sshPort = 22; $login = ''; $destination = ''; $seenPort = $false
    for ($i = 1; $i -lt $tokens.Count; $i++) {
        $token = $tokens[$i]
        if ($token -match '^-(p|l|L|i)(.*)$') {
            $option = $Matches[1]; $value = $Matches[2]
            if (-not $value) {
                $i++
                if ($i -ge $tokens.Count) { throw "Missing value for -$option." }
                $value = $tokens[$i]
            }
            switch -CaseSensitive ($option) {
                'p' {
                    if ($seenPort -or $value -notmatch '^\d{1,5}$') { throw 'Supply one numeric SSH port.' }
                    $sshPort = [int]$value; $seenPort = $true
                }
                'l' { if ($login) { throw 'SSH user was supplied more than once.' }; $login = $value }
                'L' { if ($value -notmatch '^\S+:\d+$') { throw 'Invalid -L forwarding value.' } }
                'i' { if (-not $value) { throw 'Missing -i value.' } }
            }
        } elseif ($token.StartsWith('-')) {
            throw "Unsupported SSH option '$token'. Use -HostName, -Port, -User for complex commands."
        } else {
            if ($destination) { throw 'Only one SSH destination is accepted; remote commands are not supported.' }
            $destination = $token
        }
    }
    if ($destination -match '^([^@]+)@(.+)$') {
        if ($login) { throw 'SSH user was supplied both with -l and USER@HOST.' }
        $login = $Matches[1]; $destination = $Matches[2]
    }
    if (-not $login -or -not $destination) { throw 'Supply the user explicitly as USER@HOST or -l USER HOST.' }
    Test-VastEndpoint -Name $destination -SshPort $sshPort -Login $login
}

function New-VastSshConfig {
    param([AllowEmptyString()][string]$Content, [psobject]$Endpoint, [string]$KeyPath)
    if ($Content.Contains([char]0)) { throw 'SSH config must be UTF-8 text, not UTF-16/binary.' }
    $newline = if ($Content.Contains("`r`n")) { "`r`n" } else { "`n" }
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in [regex]::Split($Content, '\r\n|\n|\r')) { $lines.Add($line) }
    $starts = @(); $blockEnd = $lines.Count
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*Host(?:\s+|\s*=\s*)(.+?)\s*(?:#.*)?$') {
            $patterns = @([regex]::Matches($Matches[1], '"[^"]*"|''[^'']*''|\S+') | ForEach-Object { $_.Value.Trim('"', "'") })
            if (@($patterns | Where-Object { $_ -ieq 'vast1' }).Count) {
                if ($patterns.Count -ne 1) { throw 'Host vast1 shares a block with other aliases. Split it into its own block before running this helper.' }
                $starts += $i
            }
        }
    }
    if ($starts.Count -gt 1) { throw 'Multiple Host vast1 blocks found; resolve them before updating.' }
    if ($starts.Count -eq 0) {
        # Prepend so generic Host * defaults cannot shadow this new alias.
        $keyForConfig = $KeyPath.Replace('\', '/')
        if ($keyForConfig -match '["\r\n]') { throw 'Identity path cannot contain quotes or line breaks.' }
        $block = @('Host vast1', "    HostName $($Endpoint.HostName)", "    Port $($Endpoint.Port)",
            "    User $($Endpoint.User)", "    IdentityFile `"$keyForConfig`"", '    IdentitiesOnly yes', '') -join $newline
        # Restore the global scope before the original file: otherwise its
        # pre-Host defaults would accidentally become options only for vast1.
        if ($Content) { return $block + $newline + 'Host *' + $newline + $Content }
        return $block + $newline
    }
    $start = $starts[0]
    # OpenSSH uses the first value obtained. An earlier wildcard/default block
    # setting endpoint fields could silently shadow this dedicated block.
    $earlierApplies = $true
    for ($i = 0; $i -lt $start; $i++) {
        if ($lines[$i] -match '^\s*Host(?:\s+|\s*=\s*)(.+?)\s*(?:#.*)?$') {
            $positive = $false; $negative = $false
            foreach ($pattern in @($Matches[1] -split '\s+')) {
                $pattern = $pattern.Trim('"', "'")
                if ($pattern.StartsWith('!')) { if ('vast1' -like $pattern.Substring(1)) { $negative = $true } }
                elseif ('vast1' -like $pattern) { $positive = $true }
            }
            $earlierApplies = $positive -and -not $negative
        } elseif ($earlierApplies -and $lines[$i] -match '^\s*(HostName|Port|User)(?:\s+|\s*=\s*)') {
            throw 'An earlier global/wildcard block can shadow Host vast1. Move the dedicated Host vast1 block above those endpoint defaults first.'
        } elseif ($lines[$i] -match '^\s*(Include|Match)(?:\s+|\s*=\s*)') {
            throw 'An Include/Match appears before Host vast1. Move its dedicated block before these directives so endpoint precedence can be verified safely.'
        }
    }
    for ($i = $start + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\s*(Host|Match)(?:\s+|\s*=\s*)') { $blockEnd = $i; break }
    }
    $newLines = [System.Collections.Generic.List[string]]::new()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -eq $start) {
            $newLines.Add($lines[$i])
            $newLines.Add("    HostName $($Endpoint.HostName)")
            $newLines.Add("    Port $($Endpoint.Port)")
            $newLines.Add("    User $($Endpoint.User)")
        } elseif ($i -gt $start -and $i -lt $blockEnd -and $lines[$i] -match '^\s*(HostName|Port|User)(?:\s+|=)') {
            continue
        } else { $newLines.Add($lines[$i]) }
    }
    $newLines -join $newline
}

# Dot sourcing loads pure helpers for offline tests; it performs no IO or SSH.
if ($MyInvocation.InvocationName -eq '.') { return }

if ($PSCmdlet.ParameterSetName -eq 'Command') {
    $endpoint = ConvertFrom-VastSshCommand -Text $Connection
} else { $endpoint = Test-VastEndpoint -Name $HostName -SshPort $Port -Login $User }

$ConfigPath = [IO.Path]::GetFullPath($ConfigPath)
$IdentityFile = [IO.Path]::GetFullPath($IdentityFile)
$BootstrapPath = [IO.Path]::GetFullPath($BootstrapPath)
$utf8 = [Text.UTF8Encoding]::new($false, $true)
[byte[]]$originalBytes = @()
if (Test-Path -LiteralPath $ConfigPath) { $originalBytes = [IO.File]::ReadAllBytes($ConfigPath) }
$hasBom = $originalBytes.Length -ge 3 -and $originalBytes[0] -eq 239 -and $originalBytes[1] -eq 187 -and $originalBytes[2] -eq 191
$offset = if ($hasBom) { 3 } else { 0 }
$content = $utf8.GetString($originalBytes, $offset, $originalBytes.Length - $offset)
$newContent = New-VastSshConfig -Content $content -Endpoint $endpoint -KeyPath $IdentityFile
$encoding = [Text.UTF8Encoding]::new([bool]$hasBom)

$scpArguments = @('-i', $IdentityFile, '-o', 'IdentitiesOnly=yes', '-o', "HostName=$($endpoint.HostName)", '-P', [string]$endpoint.Port, '--', $BootstrapPath)
$remoteHost = if ($endpoint.HostName.Contains(':')) { "[$($endpoint.HostName)]" } else { $endpoint.HostName }
$scpArguments += "$($endpoint.User)@${remoteHost}:/workspace/BOOTSTRAP.sh"
Write-Host "Target: $($endpoint.User)@$($endpoint.HostName):$($endpoint.Port) (alias vast1)"
Write-Host "Bootstrap: $BootstrapPath -> /workspace/BOOTSTRAP.sh"
if (-not $SkipUpload -and -not $WhatIfPreference) {
    foreach ($inputPath in @($IdentityFile, $BootstrapPath)) {
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) { throw "Required file not found: $inputPath" }
    }
    $scpCommand = (Get-Command scp -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
}
if ($PSCmdlet.ShouldProcess($ConfigPath, 'Update only Host vast1 without a backup')) {
    $directory = Split-Path $ConfigPath -Parent
    if (-not (Test-Path -LiteralPath $directory)) { [void][IO.Directory]::CreateDirectory($directory) }
    $temporary = Join-Path $directory ('.hope-vast-config-' + [guid]::NewGuid().ToString('N'))
    try {
        [IO.File]::WriteAllText($temporary, $newContent, $encoding)
        if (Test-Path -LiteralPath $ConfigPath) {
            # A concurrent config edit must never be overwritten silently.
            $current = [IO.File]::ReadAllBytes($ConfigPath)
            if ([Convert]::ToBase64String($current) -cne [Convert]::ToBase64String($originalBytes)) { throw 'SSH config changed while preparing; retry after reviewing it.' }
            [IO.File]::Replace($temporary, $ConfigPath, [NullString]::Value)
        } else { [IO.File]::Move($temporary, $ConfigPath) }
    } finally { if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary } }
}
if (-not $SkipUpload -and $PSCmdlet.ShouldProcess("$($endpoint.User)@$($endpoint.HostName):/workspace/BOOTSTRAP.sh", 'Upload bootstrap using SCP')) {
    & $scpCommand @scpArguments
    if ($LASTEXITCODE -ne 0) { throw "SCP failed (exit $LASTEXITCODE). SSH config remains updated; no remote setup was started." }
    Write-Host 'Uploaded. In the remote terminal run: cd /workspace && bash BOOTSTRAP.sh'
}
