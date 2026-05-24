unlink('cran/build/',recursive=TRUE,force=TRUE)
dir.create('cran/build/fru',recursive=TRUE,showWarnings=FALSE)

list.files(recursive=TRUE)->p

blocklist<-c("cranise.R","nextise.R")
setdiff(p,blocklist)->p
p[!grepl("^cran",p)]->p
p[!grepl("^\\.",p)]->p
p[!grepl("/target/",p)]->p
p[!grepl("\\.so$",p)]->p
p[!grepl("\\.o$",p)]->p

setdiff(unique(dirname(p)),'.')->d
sprintf('cran/build/fru/%s',d)->d
invisible(sapply(d,dir.create,recursive=TRUE,showWarnings=TRUE))
stopifnot(all(file.copy(p,sprintf("cran/build/fru/%s",p))))

system('git clone --branch crates-0.1.7 --single-branch https://gitlab.com/mbq/xrf cran/build/fru/src/xrf')

readLines('cran/build/fru/src/fru/Cargo.toml')->manifest
#XRF to path
manifest[grep('^xrf *=',manifest)]<-'xrf = { path = "../xrf" }'
writeLines(manifest,'cran/build/fru/src/fru/Cargo.toml')

buildignore<-c("^.git.*","\\.o$","\\.so$","\\.md$","target",".gitlab-ci.yml")
writeLines(buildignore,"cran/build/fru/.Rbuildignore")

stopifnot(file.copy('cran/configure','cran/build/fru/configure',overwrite=TRUE))
stopifnot(file.copy('cran/configure','cran/build/fru/configure.win',overwrite=TRUE))
stopifnot(file.copy('cran/Makevars','cran/build/fru/src/Makevars',overwrite=TRUE))
stopifnot(file.copy('cran/Makevars.win','cran/build/fru/src/Makevars.win',overwrite=TRUE))
