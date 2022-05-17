$files = Get-ChildItem "D:\ITSS-SharpObject\sharp_object_dataset\JPEGImages" -Filter *.jpg

$counter = 1

foreach ($f in $files){
	$oldname = $_.FullName
	$newname = "SOD_"+$counter+".jpg"
    Rename-Item $f -NewName $newname
    $newname
	$counter++
}