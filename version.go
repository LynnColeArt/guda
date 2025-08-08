// Copyright ©2019 The Gonum Authors. All rights reserved.
// Copyright ©2024 The GUDA Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package guda

import (
	"fmt"
	"runtime/debug"
)

const root = "github.com/LynnColeArt/guda"

// Version returns the version of GUDA and its checksum. The returned
// values are only valid in binaries built with module support.
//
// This function also reports the integrated compute engine (formerly Gonum)
// version information to maintain attribution and history.
//
// The exact version format returned by Version may change in future.
func Version() (version, sum string) {
	b, ok := debug.ReadBuildInfo()
	if !ok {
		return "", ""
	}
	for _, m := range b.Deps {
		if m.Path == root {
			if m.Replace != nil {
				switch {
				case m.Replace.Version != "" && m.Replace.Path != "":
					return fmt.Sprintf("%s=>%s %s", m.Version, m.Replace.Path, m.Replace.Version), m.Replace.Sum
				case m.Replace.Version != "":
					return fmt.Sprintf("%s=>%s", m.Version, m.Replace.Version), m.Replace.Sum
				case m.Replace.Path != "":
					return fmt.Sprintf("%s=>%s", m.Version, m.Replace.Path), m.Replace.Sum
				default:
					return m.Version + "*", m.Sum + "*"
				}
			}
			return m.Version, m.Sum
		}
	}
	return "", ""
}
