function Table(tbl)
  if not FORMAT:match('docx') then return tbl end
  local n = #tbl.colspecs
  local widths
  if n == 2 then widths = {0.48, 0.52}
  elseif n == 3 then
    local first = pandoc.utils.stringify(tbl.head.rows[1].cells[1])
    if first:match('Member') then widths = {0.13, 0.65, 0.22}
    else widths = {0.40, 0.30, 0.30} end
  elseif n == 4 then widths = {0.22, 0.22, 0.28, 0.28}
  elseif n == 5 then
    local first = pandoc.utils.stringify(tbl.head.rows[1].cells[1])
    if first:match('Segment') then widths = {0.29,0.13,0.13,0.20,0.25}
    elseif first:match('Skill') then widths = {0.34,0.17,0.15,0.17,0.17}
    else widths = {0.22,0.18,0.20,0.23,0.17} end
  end
  if widths then
    for i=1,n do
      local alignment = pandoc.AlignLeft
      if n >= 4 and i > 1 then alignment = pandoc.AlignCenter end
      tbl.colspecs[i] = {alignment, widths[i]}
    end
  end
  return tbl
end
