# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2022 Intel Corporation. All Rights Reserved.

target_sources(${LRS_TARGET}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/common.h"
        "${CMAKE_CURRENT_LIST_DIR}/periodic-timer.h"
        "${CMAKE_CURRENT_LIST_DIR}/stopwatch.h"
        "${CMAKE_CURRENT_LIST_DIR}/timer.h"
        "${CMAKE_CURRENT_LIST_DIR}/waiting-on.h"
        "${CMAKE_CURRENT_LIST_DIR}/work-week.h"
        "${CMAKE_CURRENT_LIST_DIR}/work-week.cpp")
