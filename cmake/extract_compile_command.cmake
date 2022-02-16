# splice in the actual commands created via CMAKE_EXPORT_COMPILE_COMMANDS
# we must do it like this and not take values from variables because generator expressions
# are expanded quite late in the build process
if (NOT COMMAND_FILE)
    message(FATAL_ERROR "No COMMAND_FILE specified!")
endif ()

if (NOT PRE_CONFIGURE_FILE)
    message(FATAL_ERROR "No PRE_CONFIGURE_FILE specified!")
endif ()

if (NOT POST_CONFIGURE_FILE)
    message(FATAL_ERROR "No POST_CONFIGURE_FILE specified!")
endif ()

if (NOT SRC_DIR_PREFIX)
    message(FATAL_ERROR "No SRC_DIR_PREFIX specified!")
endif ()

if (NOT OUT_DIR_PREFIX)
    message(FATAL_ERROR "No OUT_DIR_PREFIX specified!")
endif ()

message(STATUS "Transforming ${COMMAND_FILE}...")
file(STRINGS "${COMMAND_FILE}" COMPILE_COMMANDS REGEX "\"command\":")
list(TRANSFORM COMPILE_COMMANDS REPLACE "(^[ ]+\"command\"\\:)|(,$)" "")
list(TRANSFORM COMPILE_COMMANDS REPLACE "${SRC_DIR_PREFIX}" "<SRC>")
list(TRANSFORM COMPILE_COMMANDS REPLACE "${OUT_DIR_PREFIX}" "<OUT>")
list(JOIN COMPILE_COMMANDS "," COMPILE_COMMANDS)
configure_file("${PRE_CONFIGURE_FILE}" "${POST_CONFIGURE_FILE}")





