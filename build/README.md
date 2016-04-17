<h1> Purpose of this (seemingly) empty directory </h1>

This directory exists to keep build files separate from source tree.
The intended build process is to navigate in here and call CMake, i.e:
 
` cd "$PROJECT_BASE_DIR/build" && cmake .. && make `

However it makes no sense to check those makefiles since they are platform depended.
