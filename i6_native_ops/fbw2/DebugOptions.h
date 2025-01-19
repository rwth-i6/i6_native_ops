#ifndef _DEBUG_OPTIONS_H
#define _DEBUG_OPTIONS_H

typedef struct {
    bool     dump_edges     = false;
    bool     dump_alignment = false;
    bool     dump_output    = false;
    unsigned dump_every     = 40u;
    float    pruning        = 20.f;
    bool     explicit_merge = false;
    bool     per_frame_norm = false;
} DebugOptionsV2;

#endif  // _DEBUG_OPTIONS_H
