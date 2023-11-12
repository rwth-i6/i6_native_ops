#ifndef _DEBUG_OPTIONS_H
#define _DEBUG_OPTIONS_H

typedef struct {
    bool     dump_alignment = false;
    bool     dump_output    = false;
    unsigned dump_every     = 40u;
    float    pruning        = 10.f;
} DebugOptions;

#endif  // _DEBUG_OPTIONS_H