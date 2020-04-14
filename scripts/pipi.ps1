param (
  [Parameter(Mandatory=$true)][string]$package
)
pip install $package && 
pip freeze > requirements.txt